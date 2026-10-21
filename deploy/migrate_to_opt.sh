#!/usr/bin/env bash
# Migration de /root/forex-robot vers /opt/apps/iamrobot (clone git du depot).
#
# Usage (en root, depuis la session RDP ou MT5 tourne) :
#   curl -fsSL https://raw.githubusercontent.com/aymenkssi/forex_robot/main/deploy/migrate_to_opt.sh -o /root/migrate_to_opt.sh
#   bash /root/migrate_to_opt.sh
#
# Ce que fait le script :
#   1. sauvegarde /root/forex-robot, /opt/apps/iamrobot et les services systemd
#   2. clone le depot dans /opt/apps/iamrobot.new-<date>
#   3. y copie les secrets (config.json, users.json, mt5server.exe) et garde les
#      docker-compose.yml / nginx/default.conf actuellement en production
#   4. arrete le robot, bascule les dossiers, recree le venv
#   5. met a jour les chemins dans les services systemd, redemarre et verifie
# Le bridge MT5 n'est PAS redemarre : seul le chemin de son service est mis a
# jour (pris en compte au prochain redemarrage). La connexion MT5 est testee
# avant toute modification et avant de relancer le robot.
# En cas d'erreur apres l'arret du robot, l'ancien etat est restaure.
# /root/forex-robot n'est jamais modifie ni supprime.

set -Eeuo pipefail

OLD_DIR="${OLD_DIR:-/root/forex-robot}"
TARGET="${TARGET:-/opt/apps/iamrobot}"
REPO_URL="${REPO_URL:-https://github.com/aymenkssi/forex_robot.git}"
BRANCH="${BRANCH:-main}"
SYS_UNIT="${SYS_UNIT:-/etc/systemd/system/forex-robot.service}"
USER_UNIT="${USER_UNIT:-/root/.config/systemd/user/mt5-bridge.service}"

TS="$(date +%Y%m%d-%H%M%S)"
NEW_DIR="${TARGET}.new-${TS}"
PREV_DIR="${TARGET}.old-${TS}"
BACKUP="${BACKUP_ROOT:-/root/backups}/iamrobot-migration-${TS}"

STAGE="preflight"
BRIDGE_UNIT=0

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[!] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[X] %s\033[0m\n' "$*" >&2; exit 1; }

usr_systemctl() { XDG_RUNTIME_DIR="/run/user/$(id -u)" systemctl --user "$@"; }

# Teste MT5 a travers le bridge avec le python d'un venv : $1=python, $2=essais.
# N'appelle jamais shutdown() (cela couperait l'IPC MT5 partagee avec le robot).
mt5_check() {
    local py="$1" tries="${2:-1}" i
    for ((i = 1; i <= tries; i++)); do
        if timeout 90 "$py" - <<'PY'
import sys
from mt5linux import MetaTrader5
mt5 = MetaTrader5(host="127.0.0.1", port=18812, timeout=60)
ok = mt5.initialize()
account = mt5.account_info() if ok else None
print("  initialize =", ok, "| version =", mt5.version() if ok else None,
      "| last_error =", mt5.last_error(),
      "| compte =", getattr(account, "login", None))
sys.exit(0 if account is not None else 1)
PY
        then
            return 0
        fi
        [[ $i -lt $tries ]] && { warn "MT5 injoignable (essai $i/$tries), nouvel essai dans 10 s..."; sleep 10; }
    done
    return 1
}

# Attend que le robot ecoute sur 8001 ; echoue tout de suite si le service s'arrete.
wait_robot() {
    local i state
    for ((i = 0; i < 120; i++)); do
        ss -lnt | grep -q ":8001\b" && return 0
        state="$(systemctl show -p ActiveState --value forex-robot)"
        if [[ $i -ge 5 && ( "$state" == "inactive" || "$state" == "failed" ) ]]; then
            journalctl -u forex-robot -n 40 --no-pager || true
            return 1
        fi
        sleep 1
    done
    journalctl -u forex-robot -n 40 --no-pager || true
    return 1
}

rollback() {
    local rc=$?
    trap - ERR
    [[ "$STAGE" == "preflight" || "$STAGE" == "prepare" ]] && {
        warn "Echec avant l'arret des services : rien n'a ete modifie en production."
        rm -rf "$NEW_DIR"
        exit "$rc"
    }
    warn "ERREUR pendant l'etape '${STAGE}' : restauration de l'etat precedent..."
    systemctl stop forex-robot || true
    [[ -f "$BACKUP/forex-robot.service" ]] && cp -a "$BACKUP/forex-robot.service" "$SYS_UNIT"
    [[ -f "$BACKUP/mt5-bridge.service" ]] && cp -a "$BACKUP/mt5-bridge.service" "$USER_UNIT"
    systemctl daemon-reload || true
    [[ $BRIDGE_UNIT -eq 1 ]] && { usr_systemctl daemon-reload || true; }
    if [[ -d "$PREV_DIR" ]]; then
        [[ -d "$TARGET" ]] && mv "$TARGET" "${TARGET}.failed-${TS}"
        mv "$PREV_DIR" "$TARGET"
    fi
    systemctl start forex-robot || true
    (cd "$TARGET" && docker compose up -d --force-recreate) || true
    if wait_robot; then
        warn "Restauration terminee, robot actif depuis $OLD_DIR. Sauvegarde : $BACKUP"
    else
        warn "Restauration terminee MAIS le robot ne repond pas sur 8001."
        warn "Verifier MT5/bridge puis : systemctl restart forex-robot"
    fi
    exit "$rc"
}
trap rollback ERR

# ---------------------------------------------------------------- 1. Controles
log "Controles preliminaires"
[[ $EUID -eq 0 ]] || die "A lancer en root."
for cmd in git python3 docker ss curl tar; do
    command -v "$cmd" >/dev/null || die "Commande manquante : $cmd"
done
python3 -m venv --help >/dev/null 2>&1 || die "python3-venv manquant (apt install python3-venv)."
[[ -f "$OLD_DIR/forex_robot.py" ]] || die "$OLD_DIR/forex_robot.py introuvable."
# Sans users.json, /api/auth/setup est ouvert : n'importe qui pourrait creer l'admin.
[[ -f "$OLD_DIR/users.json" ]] || die "$OLD_DIR/users.json introuvable : migration annulee."
[[ -f "$OLD_DIR/config.json" ]] || warn "$OLD_DIR/config.json absent : valeurs par defaut au demarrage."
[[ -f "$TARGET/docker-compose.yml" ]] || die "$TARGET/docker-compose.yml introuvable."
[[ -f "$SYS_UNIT" ]] || die "$SYS_UNIT introuvable."
[[ -x "$OLD_DIR/.venv/bin/pip" ]] || die "venv introuvable : $OLD_DIR/.venv"
if [[ -f "$USER_UNIT" ]] && usr_systemctl cat mt5-bridge >/dev/null 2>&1; then
    BRIDGE_UNIT=1
else
    warn "Service utilisateur mt5-bridge non joignable : son fichier ne sera pas modifie."
fi
command -v timeout >/dev/null || die "Commande manquante : timeout"
log "Test de la connexion MT5 via le bridge (venv actuel)"
mt5_check "$OLD_DIR/.venv/bin/python" 1 \
    || die "MT5 injoignable via le bridge AVANT migration : corriger MT5/bridge d'abord (rien n'a ete modifie)."

cat <<EOF

  Source      : $OLD_DIR   (conserve tel quel)
  Destination : $TARGET     (ancien contenu -> $PREV_DIR)
  Depot       : $REPO_URL ($BRANCH)
  Sauvegarde  : $BACKUP

  Le robot sera arrete ~1 minute (le bridge MT5 n'est pas touche). Desactivez l'auto-trading
  avant de continuer (les positions ouvertes restent chez le broker).
EOF
if [[ "${1:-}" != "--yes" ]]; then
    read -r -p "Continuer ? [o/N] " answer
    [[ "$answer" =~ ^[oOyY]$ ]] || die "Annule."
fi

# --------------------------------------------------------------- 2. Sauvegarde
STAGE="prepare"
log "Sauvegarde dans $BACKUP"
mkdir -p "$BACKUP"
chmod 700 "$BACKUP"
tar -C "$(dirname "$OLD_DIR")" --exclude='.venv' --exclude='__pycache__' \
    -czf "$BACKUP/forex-robot.tar.gz" "$(basename "$OLD_DIR")"
tar -C "$(dirname "$TARGET")" -czf "$BACKUP/iamrobot.tar.gz" "$(basename "$TARGET")"
cp -a "$SYS_UNIT" "$BACKUP/forex-robot.service"
[[ $BRIDGE_UNIT -eq 1 ]] && cp -a "$USER_UNIT" "$BACKUP/mt5-bridge.service"
"$OLD_DIR/.venv/bin/pip" freeze | grep -v ' @ file://' > "$BACKUP/pip-freeze.txt"

# ----------------------------------------------------------- 3. Nouveau dossier
log "Clone du depot dans $NEW_DIR"
git clone --branch "$BRANCH" "$REPO_URL" "$NEW_DIR"

log "Copie des secrets et fichiers locaux"
for f in config.json users.json; do
    if [[ -f "$OLD_DIR/$f" ]]; then
        cp -a "$OLD_DIR/$f" "$NEW_DIR/robot/$f"
        chmod 600 "$NEW_DIR/robot/$f"
    fi
done
if [[ -d "$OLD_DIR/server" ]]; then
    # --update=none : ne pas ecraser les fichiers versionnes (mt5_server.py)
    cp -a --update=none "$OLD_DIR/server/." "$NEW_DIR/bridge/"
fi

# On garde exactement la config proxy en production ; git status montrera les ecarts.
cp -a "$TARGET/docker-compose.yml" "$NEW_DIR/docker-compose.yml"
if [[ -f "$TARGET/nginx/default.conf" ]]; then
    cp -a "$TARGET/nginx/default.conf" "$NEW_DIR/nginx/default.conf"
fi
shopt -s dotglob nullglob
for extra in "$TARGET"/*; do
    name="$(basename "$extra")"
    [[ -e "$NEW_DIR/$name" ]] || { warn "Copie de l'element local non versionne : $name"; cp -a "$extra" "$NEW_DIR/"; }
done

if ! git -C "$NEW_DIR" diff --quiet; then
    warn "Fichiers de production differents du depot (production conservee) :"
    git -C "$NEW_DIR" --no-pager diff --stat
fi
if ! diff -q "$OLD_DIR/forex_robot.py" "$NEW_DIR/robot/forex_robot.py" >/dev/null \
   || ! diff -q "$OLD_DIR/interface.html" "$NEW_DIR/robot/interface.html" >/dev/null; then
    warn "Le code du depot differe de $OLD_DIR (c'est la version du depot qui sera lancee)."
fi

# ------------------------------------------------------------------ 4. Bascule
STAGE="stop"
log "Arret du robot"
systemctl stop forex-robot

STAGE="switch"
log "Bascule : $TARGET -> $PREV_DIR, $NEW_DIR -> $TARGET"
mv "$TARGET" "$PREV_DIR"
mv "$NEW_DIR" "$TARGET"

# Le venv est cree a son emplacement final (les chemins y sont absolus).
log "Creation du venv $TARGET/.venv"
python3 -m venv "$TARGET/.venv"
"$TARGET/.venv/bin/pip" install --quiet --upgrade pip
"$TARGET/.venv/bin/pip" install --quiet -r "$BACKUP/pip-freeze.txt"
"$TARGET/.venv/bin/python" -c "import fastapi, mt5linux, numpy, pydantic, requests, uvicorn"

log "Mise a jour des chemins dans les services systemd"
rewrite_paths() {
    sed -i \
        -e "s#${OLD_DIR}/server#${TARGET}/bridge#g" \
        -e "s#${OLD_DIR}/\.venv#${TARGET}/.venv#g" \
        -e "s#${OLD_DIR}#${TARGET}/robot#g" \
        "$1"
    if grep -q "$OLD_DIR" "$1"; then
        echo "Chemin $OLD_DIR encore present dans $1" >&2
        return 1
    fi
}
rewrite_paths "$SYS_UNIT"
diff -u "$BACKUP/forex-robot.service" "$SYS_UNIT" || true
systemctl daemon-reload
if [[ $BRIDGE_UNIT -eq 1 ]]; then
    rewrite_paths "$USER_UNIT"
    diff -u "$BACKUP/mt5-bridge.service" "$USER_UNIT" || true
    usr_systemctl daemon-reload
fi

# --------------------------------------------------------------- 5. Redemarrage
STAGE="start"
log "Test de la connexion MT5 via le bridge (nouveau venv)"
mt5_check "$TARGET/.venv/bin/python" 6

log "Demarrage du robot"
systemctl start forex-robot
wait_robot
curl -fsS http://127.0.0.1:8001/api/auth/status >/dev/null

log "Recreation du conteneur iamrobot-proxy depuis $TARGET"
(cd "$TARGET" && docker compose up -d --force-recreate)
sleep 3
docker exec iamrobot-proxy wget -qO- http://host.docker.internal:8001/api/auth/status >/dev/null

STAGE="done"
trap - ERR
log "Migration terminee"
cat <<EOF

  Projet         : $TARGET  (git -C $TARGET status)
  Ancien proxy   : $PREV_DIR
  Ancien robot   : $OLD_DIR  (non modifie)
  Sauvegarde     : $BACKUP

  Verifier : https://iamrobot.me puis   journalctl -u forex-robot -n 50 --no-pager
  Apres quelques jours sans probleme :  rm -rf $PREV_DIR $OLD_DIR

  Mises a jour futures :  cd $TARGET && git pull && systemctl restart forex-robot
EOF

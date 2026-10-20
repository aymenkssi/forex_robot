# Forex Robot Pro — iamrobot.me

Robot de trading Forex pour MetaTrader 5 (sous Wine) avec analyse technique multi-paires,
moteur de risque ATR, filtre calendrier économique, validation par IA locale (Ollama/Mistral),
alertes Telegram et interface web authentifiée.

## Architecture

```
Internet ─443─> Traefik (Docker) ─> iamrobot-proxy (Nginx Docker) ─> host.docker.internal:8001
                                                                        │
                                              FastAPI  robot/forex_robot.py
                                               │                         │
                              bridge mt5linux/RPyC 127.0.0.1:18812   Ollama 127.0.0.1:11434
                                               │
                                MetaTrader 5 sous Wine (WINEPREFIX=/root/.mt5)
```

## Arborescence (= `/opt/apps/iamrobot` sur le VPS)

| Chemin | Contenu |
|---|---|
| `robot/forex_robot.py` | API FastAPI + auto-trader |
| `robot/interface.html` | Interface web servie sur `/` |
| `robot/config.example.json` | Modèle de `config.json` (sans secrets) |
| `robot/requirements.txt` | Dépendances Python Linux |
| `bridge/mt5_server.py` | Serveur RPyC côté Wine (alternative au module `mt5linux`) |
| `docker-compose.yml`, `nginx/default.conf` | Proxy publié par Traefik |
| `deploy/systemd/` | Modèles des services `forex-robot` et `mt5-bridge` |
| `deploy/migrate_to_opt.sh` | Migration `/root/forex-robot` → `/opt/apps/iamrobot` |

Non versionnés (voir `.gitignore`) : `robot/config.json` (token Telegram),
`robot/users.json` (comptes de l'interface), `bridge/*.exe`, `.venv/`.

> ⚠️ Ce dépôt est **public** : ne jamais y committer de mot de passe, token Telegram,
> identifiant MT5 ou jeton de session.

## Migration vers /opt/apps/iamrobot

En root sur le VPS, depuis la session RDP où MT5 tourne, **auto-trading désactivé** :

```bash
curl -fsSL https://raw.githubusercontent.com/aymenkssi/forex_robot/main/deploy/migrate_to_opt.sh -o /root/migrate_to_opt.sh
bash /root/migrate_to_opt.sh
```

Le script sauvegarde tout dans `/root/backups/iamrobot-migration-<date>/`, clone le dépôt,
recopie les secrets et la configuration proxy en production, recrée le venv, met à jour les
chemins des services systemd puis redémarre et vérifie bridge, robot et proxy.
En cas d'échec après l'arrêt des services, il restaure automatiquement l'état précédent.
`/root/forex-robot` n'est pas modifié : à supprimer manuellement après quelques jours.

## Mise à jour après migration

```bash
cd /opt/apps/iamrobot
git pull
systemctl restart forex-robot
# si nginx/default.conf ou docker-compose.yml ont changé :
docker compose up -d --force-recreate
```

## Vérifications

```bash
ss -lntp | grep -E ':8001|:18812'
systemctl status forex-robot --no-pager
systemctl --user status mt5-bridge --no-pager
journalctl -u forex-robot -n 50 --no-pager
docker exec iamrobot-proxy wget -qO- http://host.docker.internal:8001/api/auth/status
```

import MetaTrader5 as mt5
from rpyc.cli.rpyc_classic import ClassicServer


print("=== Initialisation MetaTrader 5 ===")

if not mt5.initialize():
    print("ERREUR: MT5.initialize() a échoué")
    print("last_error =", mt5.last_error())
    raise SystemExit(1)

print("MT5 initialise avec succès")
print("Terminal :", mt5.terminal_info())
print("Compte   :", mt5.account_info())

print("=== Démarrage serveur RPyC ===")

ClassicServer.run()

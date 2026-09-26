import spade
from swarm.main import main

if __name__ == "__main__":
    spade.run(main(), embedded_xmpp_server=True)

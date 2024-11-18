from paper_trading import PaperTrader
from config import config

def main():
    pt = PaperTrader(config)
    pt.execute_trade()

def test_main():
    pt = PaperTrader(config, debug=True)
    pt.test_execute_trade()

if __name__ == '__main__':

    main()
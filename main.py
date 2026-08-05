# python -m src.main
import logging
from src.config import config
from src.app import Application

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    app = Application(config)
    app.run(show_window=True)


if __name__ == "__main__":
    main()

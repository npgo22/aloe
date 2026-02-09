from . import gui  # noqa: F401
from nicegui import ui


def main():
    ui.run(host="0.0.0.0", port=8080, title="Aloe: Nolan's Sim", reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()

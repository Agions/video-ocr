import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from visionsub.services.batch_service import BatchService
from visionsub.ui.main_window import MainWindow
from visionsub.view_models.main_view_model import MainViewModel


def main():
    """The main entry point for the application."""
    app = QApplication(sys.argv)

    # --- Dependency Injection ---
    # 1. Create the services
    batch_service = BatchService()
    # 2. Create the ViewModel and inject services
    main_view_model = MainViewModel(batch_service)
    # 3. Create the View and inject the ViewModel
    main_window = MainWindow(main_view_model)

    # --- Load Stylesheet ---
    try:
        style_path = Path(__file__).parent / "ui/style.qss"
        with open(style_path, "r") as f:
            app.setStyleSheet(f.read())
    except FileNotFoundError:
        print("Warning: Stylesheet 'style.qss' not found. Using default style.")

    # --- Show the main window and run the application ---
    main_window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

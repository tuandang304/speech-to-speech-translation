import webbrowser
from threading import Timer
from webapp.backend.app import app

def open_browser():
    """
    Opens the default web browser to the webapp's URL.
    """
    webbrowser.open_new("http://127.0.0.1:5001/")

def main():
    """
    Main function to run the Flask app.
    """
    # Open the browser 1 second after the server starts
    Timer(1, open_browser).start()
    # Run the Flask app
    app.run(host='127.0.0.1', port=5001)

if __name__ == '__main__':
    main()
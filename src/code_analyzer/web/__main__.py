from code_analyzer.web.server import run_server

def main():
    """Start the web interface."""
    run_server(port=8000, open_browser=True)

if __name__ == '__main__':
    main()
from app import create_app

app = create_app()

if __name__ == "__main__":
    # Use a non-default port in case 5000 is occupied (macOS AirPlay sometimes uses 5000)
    app.run(host="0.0.0.0", port=5001, debug=True)

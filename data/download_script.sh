#!/bin/bash

# Check if Google Chrome is installed
if ! command -v google-chrome >/dev/null 2>&1; then
    echo "Google Chrome is not installed. Installing it now..."
    # Install Google Chrome
    sudo apt-get update
    sudo apt-get install -y curl
    wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
    sudo apt install -y ./google-chrome-stable_current_amd64.deb
else
    echo "Google Chrome is already installed."
fi

# Check if ChromeDriver is installed
if ! command -v chromedriver >/dev/null 2>&1; then
    echo "ChromeDriver is not installed. Installing it now..."
    # Download and install the latest ChromeDriver
    wget https://chromedriver.storage.googleapis.com/114.0.5735.90/chromedriver_linux64.zip  # Replace this URL with the required version if necessary
    unzip chromedriver_linux64.zip
    sudo mv chromedriver /usr/bin/chromedriver
    sudo chown root:root /usr/bin/chromedriver
    sudo chmod +x /usr/bin/chromedriver
else
    echo "ChromeDriver is already installed."
fi

# After necessary installations, download the data using python script
python data/download_data.py

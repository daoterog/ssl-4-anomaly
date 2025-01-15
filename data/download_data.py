"""This script downloads data from the SewerML website. Only new files are downloaded to avoid
duplicates and deal with download errors."""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


def initialize_chrome_driver(download_dir):
    """Initialize the Chrome driver with the download directory
    settings and return the driver object."""
    # Initialize the Chrome driver with the download directory settings
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_experimental_option(
        "prefs",
        {
            "download.default_directory": str(
                download_dir
            ),  # Set the download directory
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
        },
    )

    # Initialize the Chrome driver
    service = Service(ChromeDriverManager().install())
    # return the driver object
    return webdriver.Chrome(service=service, options=chrome_options)


def login(driver):
    """Login to the SewerML website using the provided credentials."""
    # Head to landing page
    driver.get(os.environ.get("SEWERML_URL"))

    # Find password input field and insert password as well
    driver.find_element(By.ID, "password").send_keys(os.environ.get("SEWERML_PASSWORD"))

    # Find the login button and click it
    driver.find_element(
        By.CSS_SELECTOR, "input.svg.icon-confirm[type='submit']"
    ).click()
    # Wait for a specific element that indicates a successful login
    try:
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.ID, "select_all_files"))
        )
        print(" --- Login successful --- ")
    except Exception as e:  # pylint: disable=broad-except
        print("Login failed:", str(e))
        driver.quit()


def wait_for_all_downloads_to_complete(directory, timeout=3600):
    """Wait until all ongoing downloads (indicated by .crdownload files) are complete."""
    end_time = time.time() + timeout
    while time.time() < end_time:
        # Check for any ongoing downloads (.crdownload files)
        ongoing_downloads = [
            f for f in os.listdir(directory) if f.endswith(".crdownload")
        ]

        if not ongoing_downloads:
            # No ongoing downloads, proceed
            return True

        time.sleep(5)  # Check every 5 seconds

    print("Download did not complete within the timeout period.")
    return False


# Scroll until all checkboxes are visible
def scroll_to_load_elements(driver, css_selector):
    """Scroll down to load all elements matching the provided CSS selector."""
    # Scroll down and check for checkboxes
    previous_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Find checkboxes
        checkboxes = driver.find_elements(By.CSS_SELECTOR, css_selector)
        if checkboxes:
            print(f"Found {len(checkboxes)} checkboxes.")

        # Scroll down
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Allow time for new elements to load

        # Check the new height of the page
        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == previous_height:
            # No new elements loaded
            break
        previous_height = new_height


def list_all_checkboxes(driver):
    """List all checkboxes on the page except the 'select all' checkbox. To ensure a single
    file is downloaded at a time (as a checkpointing technique)."""
    # Scroll to load all checkboxes
    scroll_to_load_elements(driver, "input[type='checkbox']")

    # Find all checkboxes on the page
    checkboxes = driver.find_elements(By.CSS_SELECTOR, "input[type='checkbox']")

    # Filter out the checkbox with id="select_all_files"
    checkboxes = [
        checkbox
        for checkbox in checkboxes
        if checkbox.get_attribute("id") != "select_all_files"
    ]

    # Check if the number of checkboxes is as expected
    assert len(checkboxes) == 24, "Number of checkboxes is not as expected."

    return checkboxes


def list_local_files(directory):
    """List all files in the provided directory."""
    return [
        os.path.basename(f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and not f.endswith(".crdownload")
    ]


def download_data_from_webpage(driver, download_dir):
    """Download data from the SewerML website. The function iterates through each checkbox
    on the page, selects it, and initiates the download process. Only new files are downloaded
    to avoid duplicates."""
    # list all checkboxes available
    checkboxes = list_all_checkboxes(driver)

    # get file names associated with each checkbox
    file_names = {}
    for checkbox in checkboxes:
        # Find the associated file name
        parent_div = checkbox.find_element(By.XPATH, "..").find_element(By.XPATH, "..")
        file_name_span = parent_div.find_element(By.CSS_SELECTOR, "span.innernametext")
        file_name = file_name_span.text
        file_names[checkbox] = file_name

    # Scroll back up to top and wait 2 secs to start the download
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(2)

    # get local files to avoid downloading the same files
    local_files = list_local_files(download_dir)
    # delete files extensions
    local_files = [f.split(".")[0] for f in local_files]

    # Calculate the initial file count before starting the download process
    initial_file_count = len(local_files)

    # Keep track of the previously selected checkbox
    previous_checkbox = None

    # Process each checkbox one by one
    for checkbox in checkboxes:
        if file_names[checkbox] in local_files:
            print(
                f"File {file_names[checkbox]} already exists locally. \033[93mSkipping.\033[0m"
            )
            continue

        try:
            # Ensure only one checkbox is selected at a time
            if checkbox.is_selected():
                print(f"Checkbox {checkbox.get_attribute('id')} is already selected.")
            else:
                # If there's a previously selected checkbox, unselect it
                if previous_checkbox and previous_checkbox.is_selected():
                    previous_checkbox.click()
                    print(
                        f"Checkbox {previous_checkbox.get_attribute('id')} unselected."
                    )

                # Wait 2 seconds before clicking
                time.sleep(2)

                # Click the current checkbox
                checkbox.click()
                print(f"Checkbox {checkbox.get_attribute('id')} selected.")

                # Scroll back to the top to click the download button
                driver.execute_script("window.scrollTo(0, 0);")
                time.sleep(2)  # Allow time for the scroll to complete

                # Find the download button and click it
                download_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable(
                        (By.CSS_SELECTOR, "a.download.btn.btn-xs.btn-default")
                    )
                )
                download_button.click()
                print(
                    f"\033[94mDownload initiated for file {file_names[checkbox]}.\033[0m"
                )
                # Add sleep time to avoid double downloads
                time.sleep(5)

                # Wait for the download to complete
                if wait_for_all_downloads_to_complete(download_dir):
                    print(
                        f"Download for file {file_names[checkbox]} \033[92mcompleted\033[0m."
                    )

                    # Update the file count for the next iteration
                    initial_file_count += 1

                else:
                    print(
                        f"Download for file {file_names[checkbox]} did not complete within\
                            the timeout period."
                    )

            # update previous checkbox
            previous_checkbox = checkbox
            # Wait for 3 seconds before processing the next checkbox
            time.sleep(3)

        except Exception as e:  # pylint: disable=broad-except
            print(
                f"\033[91mAn error occurred for file {file_names[checkbox]}: {str(e)}\033[0m"
            )


if __name__ == "__main__":
    # Define current directory and download directory
    CURR_DIR = Path(__file__).resolve().parent
    DOWNLOAD_DIR = CURR_DIR / "sewer-ml"

    # Create the download directory if it doesn't exist
    if not DOWNLOAD_DIR.exists():
        DOWNLOAD_DIR.mkdir()

    # Load environment variables
    load_dotenv(CURR_DIR.parent / ".env", override=True)
    # initialize driver
    chrome_driver = initialize_chrome_driver(DOWNLOAD_DIR)
    # Login to the SewerML website
    login(chrome_driver)
    # Download data from the SewerML website
    download_data_from_webpage(chrome_driver, DOWNLOAD_DIR)

    # Close the driver
    chrome_driver.quit()

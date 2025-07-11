import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

def setup_driver():
    """Sets up the Selenium WebDriver for Chrome."""
    # To use this script, you need to have Google Chrome and
    # the corresponding version of ChromeDriver installed.
    # For automated driver management, you could use webdriver-manager.
    # pip install webdriver-manager
    # from selenium.webdriver.chrome.service import Service as ChromeService
    # from webdriver_manager.chrome import ChromeDriverManager
    # driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    
    # Using basic setup for simplicity. Assumes chromedriver is in PATH.
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless')  # Uncomment to run in the background
    try:
        driver = webdriver.Chrome(options=options)
        return driver
    except Exception as e:
        print("Error setting up WebDriver. Make sure Chrome and ChromeDriver are installed and in your PATH.")
        print(f"Error details: {e}")
        return None

def get_four_corner_codes_batch(driver, chars):
    """
    Queries the four-corner codes for a batch of Chinese characters using Selenium.
    """
    url = "https://www.qncha.com/sijiao/"
    codes = {}
    try:
        driver.get(url)
        input_xpath = "/html/body/div[2]/div[1]/div[1]/div[2]/form/span[1]/input"
        input_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, input_xpath))
        )
        input_box.clear()
        input_box.send_keys("".join(chars))

        button_xpath = "/html/body/div[2]/div[1]/div[1]/div[2]/form/span[2]/input"
        search_button = driver.find_element(By.XPATH, button_xpath)
        search_button.click()

        result_xpath = "/html/body/div[2]/div[1]/div[2]/div[3]/p[4]"
        result_container = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, result_xpath))
        )

        links = result_container.find_elements(By.TAG_NAME, 'a')
        for link in links:
            link_text = link.text.strip()
            if len(link_text) > 1:
                char = link_text[-1]
                try:
                    span = link.find_element(By.TAG_NAME, 'span')
                    code = span.text.strip()
                    if code and char in chars:
                        codes[char] = code
                except NoSuchElementException:
                    continue
        return codes

    except (TimeoutException, NoSuchElementException):
        print(f"  -> Timeout or element not found while searching for batch starting with '{chars[0]}'.")
        return {}
    except Exception as e:
        print(f"  -> An unexpected error occurred for batch starting with '{chars[0]}': {e}")
        return {}

def main():
    """
    Main function to read characters from a CSV, scrape their four-corner codes,
    and write the results back to the CSV.
    """
    csv_file = 'four_corner_data.csv'

    # Try to read the CSV file
    try:
        df = pd.read_csv(csv_file, dtype={'character': str, 'four_corner': str})
        print(f"Successfully loaded {len(df)} entries from {csv_file}.")
    except FileNotFoundError:
        print(f"File '{csv_file}' not found. Creating a new one with sample characters.")
        char_list_str = "的一是在不了有和人这中大为上个国我以要他时来用们生动社会地说种"
        char_list = sorted(list(set(list(char_list_str))))
        df = pd.DataFrame(char_list, columns=['character'])
        df['four_corner'] = ''
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"Sample file created. Please populate '{csv_file}' with desired characters.")
        # Reread the newly created file
        df = pd.read_csv(csv_file, dtype={'character': str, 'four_corner': str})

    # Ensure 'four_corner' column exists, if not, add it.
    if 'four_corner' not in df.columns:
        df['four_corner'] = ''

    # Setup WebDriver
    driver = setup_driver()
    if not driver:
        return # Exit if driver setup failed

    # Find characters that need their four-corner code
    to_process = df[(df['character'].str.len() == 1) & 
                    (df['character'].str.match(r'[\u4e00-\u9fff]')) & 
                    (df['four_corner'].isna() | (df['four_corner'] == ''))]

    chars_to_query = to_process['character'].tolist()
    if not chars_to_query:
        print("No new characters to process.")
        driver.quit()
        return

    print(f"Found {len(chars_to_query)} characters to query.")

    # Process characters in batches of 50
    batch_size = 50
    updated_count = 0
    for i in range(0, len(chars_to_query), batch_size):
        batch_chars = chars_to_query[i:i+batch_size]
        print(f"Querying batch {i//batch_size + 1}... ({len(batch_chars)} chars)")
        codes = get_four_corner_codes_batch(driver, batch_chars)
        
        if codes:
            for char, code in codes.items():
                df.loc[df['character'] == char, 'four_corner'] = code
                print(f"  -> Found code for '{char}': {code}")
                updated_count += 1
        else:
            print("  -> No codes found for this batch.")
        
        time.sleep(1)  # Be polite to the server


    # Quit the driver
    driver.quit()

    # Save the updated DataFrame to CSV if any changes were made
    if updated_count > 0:
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"\nFinished. Updated {updated_count} entries in {csv_file}.")
    else:
        print("\nFinished. No new codes were found or updated.")

if __name__ == "__main__":
    main()

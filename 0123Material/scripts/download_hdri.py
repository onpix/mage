import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import os


def download_image(url, folder="files/hdr"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    response = requests.get(url)
    if response.status_code == 200:
        image_name = url.split("/")[-1]
        with open(os.path.join(folder, image_name), "wb") as file:
            file.write(response.content)


def process_page(page):
    print(f"Downloading page {page}...")
    url = f"https://hdri-haven.com/?page={page}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    hdri_divs = soup.find_all("div", class_="hdri")

    for div in hdri_divs:
        img_tag = div.find("img")
        if img_tag and "src" in img_tag.attrs:
            thumb_url = img_tag["src"]
            image_file = thumb_url.split("/")[-1]
            # Construct the 4K HDRI download link
            download_url = f"https://hdri.top/hdris/4k/{image_file.replace('.jpg', '_4k.hdr')}"
            download_image(download_url)


def scrape_hdri_haven(start_page, end_page, max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_page, range(start_page, end_page + 1))


# Example usage
scrape_hdri_haven(1, 34)  # Scrape pages 1 to 5 as an example

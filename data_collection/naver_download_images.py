#%%
import pandas as pd
import requests
from pathlib import Path
import mimetypes
import pickle
import re
from PIL import Image 
from io import BytesIO
from tqdm import tqdm

def download_image(image_url: str, destination_directory: Path, base_filename: str | None = None) -> Path:
    """
    Downloads an image from a given URL to a specified directory, inferring
    the file extension from the Content-Type header if possible.
    If the image is larger than 320x320, it will be proportionally resized to fit within 320x320.

    Args:
        image_url (str): The URL of the image to download.
        destination_directory (Path): The Path object representing the
                                     directory where the image will be saved.
        base_filename (str, optional): The desired base name for the downloaded file
                                       (without extension). If None, the base name
                                       will be derived from the image_url. Defaults to None.

    Returns:
        Path: The Path object representing the local path of the downloaded image.

    Raises:
        requests.exceptions.RequestException: If there's an issue with the
                                              HTTP request (e.g., network error,
                                              invalid URL, non-200 status code).
        IOError: If there's an issue writing the image to the file system or processing.
        ValueError: If a suitable file extension cannot be determined or
                    if the URL is invalid.
    """
    MAX_SIZE = (320, 320) # Define the maximum dimensions for the image

    try:
        # Ensure the destination directory exists
        destination_directory.mkdir(parents=True, exist_ok=True)

        # 1. Determine base filename from input or URL, with sanitization
        if base_filename:
            if not base_filename:
                raise ValueError("Provided base_filename is empty or invalid after sanitization.")
        else:
            base_filename = Path(image_url).stem
            if not base_filename:
                base_filename = "downloaded_file"

        # 2. Try to infer extension using a HEAD request (most efficient for initial guess)
        inferred_extension = None
        try:
            with requests.head(image_url, allow_redirects=True, timeout=10) as head_response:
                head_response.raise_for_status()
                content_type = head_response.headers.get('Content-Type')
                if content_type:
                    inferred_extension = mimetypes.guess_extension(content_type.split(';')[0].strip(), strict=False)
                    if inferred_extension == '.jpe':
                        inferred_extension = '.jpeg'
        except requests.exceptions.RequestException as e:
            print(f"Warning: HEAD request failed for {image_url}. {e}. Will attempt to infer from GET response.")

        # 3. Proceed with GET request to download image content
        # Store content in memory first for Pillow processing
        image_content = BytesIO() # Use BytesIO to accumulate image data
        with requests.get(image_url, stream=True, allow_redirects=True, timeout=30) as response:
            response.raise_for_status()

            # Ensure we have a valid Content-Type before trying to process as image
            get_content_type = response.headers.get('Content-Type')
            if not get_content_type or not get_content_type.startswith('image/'):
                raise ValueError(f"URL {image_url} does not appear to serve an image (Content-Type: {get_content_type})")

            # Update inferred_extension if HEAD failed or was less accurate
            if not inferred_extension:
                if get_content_type:
                    inferred_extension = mimetypes.guess_extension(get_content_type.split(';')[0].strip(), strict=False)
                    if inferred_extension == '.jpe':
                        inferred_extension = '.jpeg'

            if not inferred_extension:
                url_suffix = Path(image_url).suffix.lower()
                if url_suffix and url_suffix in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg']:
                    inferred_extension = url_suffix
                    print(f"Warning: No Content-Type extension found. Using URL suffix '{url_suffix}'.")
                else:
                    raise ValueError(f"Could not determine a suitable file extension for {image_url}. "
                                     "Content-Type header was missing or unrecognized, and URL suffix was not definitive.")

            for chunk in response.iter_content(chunk_size=8192):
                image_content.write(chunk)

        # 4. Process image with Pillow
        image_content.seek(0) # Rewind the BytesIO object to the beginning
        try:
            with Image.open(image_content) as img:
                original_width, original_height = img.size
                print(f"Original image size: {original_width}x{original_height}")

                # Check if resizing is needed
                if original_width > MAX_SIZE[0] or original_height > MAX_SIZE[1]:
                    img.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS) # Use thumbnail for proportional resizing
                    print(f"Resized image to: {img.size[0]}x{img.size[1]}")
                else:
                    print("Image is within size limits, no resizing needed.")

                # Determine file format for saving based on inferred extension
                # Pillow often infers format from extension or can be explicit
                save_format = None
                if inferred_extension:
                    if inferred_extension == '.jpg':
                        save_format = 'JPEG'
                    elif inferred_extension == '.png':
                        save_format = 'PNG'
                    elif inferred_extension == '.gif':
                        save_format = 'GIF'
                    elif inferred_extension == '.webp':
                        save_format = 'WEBP'
                    # Add more formats as needed. If None, Pillow tries to infer.
                
                # Construct the final local path
                local_image_path = destination_directory / (base_filename + inferred_extension)

                # Ensure filename is unique
                original_local_image_path = local_image_path
                counter = 0
                while local_image_path.exists():
                    counter += 1
                    local_image_path = destination_directory / f"{base_filename}_{counter}{inferred_extension}"
                    if counter > 100:
                        raise IOError(f"Could not find a unique filename for {original_local_image_path.name} after {counter-1} attempts.")

                # Save the processed image
                # quality parameter is only for JPEG/WebP.
                # optimize=True can reduce file size for PNG/JPEG without quality loss
                if save_format == 'JPEG':
                    img.save(local_image_path, format=save_format, quality=85, optimize=True)
                elif save_format == 'PNG':
                     img.save(local_image_path, format=save_format, optimize=True)
                else:
                    img.save(local_image_path, format=save_format) # Let Pillow decide format/quality for others
                
        except Exception as img_exc:
            raise IOError(f"Error processing image with Pillow from {image_url}: {img_exc}")

        print(f"Image downloaded and processed successfully to: {local_image_path}")
        return local_image_path

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image from {image_url}: {e}")
        return Path("")
    except IOError as e:
        print(f"Error saving image: {e}")
        return Path("")
    except ValueError as e:
        print(f"File handling error: {e}")
        return Path("")
    except Exception as e:
        print(f"An unexpected error occurred during download_image: {e}")
        return Path("")
def convert_blog_url_to_dir(blog_url:str):
    blog_begin = re.compile("https://m.blog.naver.com/")
    a = re.sub(blog_begin, "", blog_url)
    b = re.sub("/", "-", a)
    return b
def convert_dir_to_blog_url(dir:str):
    a = re.sub("-", "/", dir)
    b = "https://m.blog.naver.com/" + a
    return b

def main(datadir_path=Path("G:/My Drive/Data/naver_search_results/"),
         imagedir = Path(__file__).parent.parent.parent / "images"):
    navermap_reviews_path = datadir_path / "navermap_reviews.parquet"
    naverblog_reviews_path = datadir_path / "naverblog_reviews.parquet"
    navermap_reviews = pd.read_parquet(navermap_reviews_path, 
                                       engine="pyarrow", 
                                       columns=["review_id", "image_links", "video_thumbnail_links"])
    map_imagedir = imagedir / "map"
    map_image_paths = {}
    for all_i, map_series in tqdm(navermap_reviews.iterrows()):
        print(f"{all_i+1}/ {len(navermap_reviews)}")
        review_id = map_series["review_id"]
        map_image_paths[review_id] = {}
        map_image_paths[review_id]["image"] = []
        map_image_paths[review_id]["vid"] = []
        destination_dir = map_imagedir / review_id
        img_list = map_series["image_links"]
        if img_list is not None and len(img_list) > 0:
            all_img = len(img_list)
            for i, img_url in enumerate(img_list):
                base_filename = review_id + "_" + str(i)
                to_path = download_image(image_url=img_url, 
                                         destination_directory=destination_dir,
                                         base_filename=base_filename)
                map_image_paths[review_id]["image"].append(to_path)
                print(f"{i+1}/{all_img} | Downloaded: {to_path}")
        vid_list = map_series["video_thumbnail_links"]
        if vid_list is not None and len(vid_list) > 0:
            all_vid = len(vid_list)
            for i, vid_url in enumerate(vid_list):
                base_filename = review_id + "_vid" + str(i)
                to_path = download_image(image_url=vid_url,
                                         destination_directory=destination_dir,
                                         base_filename=base_filename)
                map_image_paths[review_id]["vid"].append(to_path)
                print(f"{i+1}/{all_vid} | Downloaded: {to_path}")
    with open(datadir_path/"navermap_reviews_image_local.pkl") as wf:
        pickle.dump(map_image_paths, wf)
    del navermap_reviews
    blog_imagedir = imagedir / "blog"
    naverblog_reviews = pd.read_parquet(naverblog_reviews_path, 
                                        engine="pyarrow",
                                        columns = ["post_url", "img_url", "sticker_url", "vid_thumb_url"])
    blog_len = len(naverblog_reviews)
    blog_image_paths = {}
    for all_i, blog_series in tqdm(naverblog_reviews.iterrows()):
        print(f"{all_i + 1}/{blog_len}")
        blog_url = blog_series["post_url"]
        blog_image_paths[blog_url] = {}
        blog_image_paths[blog_url]["image"] = []
        blog_image_paths[blog_url]["sticker"] = []
        blog_image_paths[blog_url]["vid"] = []
        blog_folder = convert_blog_url_to_dir(blog_url)
        destination_dir = blog_imagedir / blog_folder
        img_list = blog_series["img_url"]
        sticker_list = blog_series["sticker_url"]
        vid_list = blog_series["vid_thumb_url"]
        if img_list is not None and len(img_list) > 0:
            all_img = len(img_list)
            for i, img_url in enumerate(img_list):
                base_filename = blog_folder + "_" + str(i)
                to_path = download_image(img_url,
                                         destination_directory=destination_dir,
                                         base_filename=base_filename)
                blog_image_paths[blog_url]["image"].append(to_path)
                print(f"{i+1}/{all_img} | Downloaded: {to_path}")
        if sticker_list is not None and len(sticker_list) > 0:
            all_sticker = len(sticker_list)
            for i, sticker_url in enumerate(sticker_list):
                base_filename = blog_folder + "_sticker" + str(i)
                to_path = download_image(sticker_url,
                                         destination_directory=destination_dir,
                                         base_filename=base_filename)
                blog_image_paths[blog_url]["sticker"].append(to_path)
                print(f"{i+1}/{all_sticker} | Downloaded: {to_path}")
        if vid_list is not None and len(vid_list) > 0:
            all_vid = len(vid_list)
            for i, vid_url in enumerate(vid_list):
                base_filename = blog_folder + "_vid" + str(i)
                to_path = download_image(vid_url,
                                         destination_directory=destination_dir,
                                         base_filename=base_filename)
                blog_image_paths[blog_url]["vid"].append(to_path)
                print(f"{i+1}/{all_vid} | Downloaded: {to_path}")
    del naverblog_reviews

if __name__ == "__main__":
    main()

import gdown

def download(url, model_name):
    """
    Download weights from Google Drive

    Parameters
    ----------
    url : str
        The url of the file to download

    Returns
    -------
    output : str
        The name of the downloaded file

    """

    output_path = 'models/' +  model_name + '/weights.h5'
    print("Downloading weights from Google Drive...")
    gdown.download(url, output= output_path, quiet=False, fuzzy = True)
    print("Download completed")
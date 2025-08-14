import gdown

# Google Drive file ID
file_id = "1DCok-c2Tbet4rDpHO2-yQ3bJ2RlP4VSm"
url = f"https://drive.google.com/uc?id={file_id}"

# Output filename
output = "embeddings.pkl"

print("Downloading embeddings.pkl from Google Drive...")
gdown.download(url, output, quiet=False)
print("Download complete!")

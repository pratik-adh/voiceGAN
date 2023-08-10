import os

def check_if_file_exists(filepath):

    # Check if the file exists.
    if os.path.exists(filepath):
        return True
    else:
        return False
    
def get_foldername(mp3_folder):

    # Get the filename from the path.
    foldername = os.path.basename(mp3_folder)

    return foldername

def create_parent_and_child_directory(parent_directory, child_directory):
  if not os.path.exists(parent_directory):
    os.mkdir(parent_directory)
  child_directory_path = os.path.join(parent_directory, child_directory)
  if not os.path.exists(child_directory_path):
    os.mkdir(child_directory_path)
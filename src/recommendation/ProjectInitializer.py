import os
from options.AllOptions import AllOptions

if __name__ == "__main__":
    data_paths = [AllOptions.DataOptions.raw_folder_path,
                  AllOptions.DataOptions.tmp_folder_path,
                  AllOptions.DataOptions.cache_folder_path,
                  AllOptions.DataOptions.graphs_folder_path,
                  AllOptions.DataOptions.checkpoints_folder_path]
    
    for path in data_paths:
        if not os.path.exists(path):
            os.makedirs(path)

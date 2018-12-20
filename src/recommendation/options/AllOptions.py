from options.ModelOptions import ModelOptions
from options.DataOptions import DataOptions

class AllOptions():
    '''Options for the user to manipulate.
    
    root_path: Where the recommendation folder is located.
    DataOptions: Edit the values in this class to change where data is stored.
    ModelOptions: Edit the values in this class to change what the model looks like and how it is trained.
    '''
    
    root_folder_path = "./src/recommendation/"
    DataOptions = DataOptions(root_folder_path)
    ModelOptions = ModelOptions()
    

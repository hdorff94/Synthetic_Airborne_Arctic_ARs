# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:40:41 2020

@author: u300737
"""
import os
from configparser import ConfigParser
from termcolor import colored

def create_new_config_file(file_name="data_config.ini"):
    import sys
    #Get the configparser object
    config_object= ConfigParser()
    system_is_windows=sys.platform.startswith("win")
    if not system_is_windows:
        
        config_object["Data_Paths"]={
        "system":"linux",
        "campaign":"NAWDEX",
        "campaign_path":"/scratch/uni/u237/users/hdorff/",
            }
        config_object["Data_Paths"]["save_path"]="/home/zmaw/u300737/PhD/Work/"+\
        config_object["Data_Paths"]["campaign"]
    else:
        config_object["Data_Paths"]={"system":"windows",
                                    "campaign":"NAWDEX",
                                    "campaign_path":os.getcwd()+"/"}    
        config_object["Data_Paths"]["save_path"]=\
            config_object["Data_Paths"]["campaign_path"]+"Save_path/"
        
        
        #        self.add_entries_to_config_object(major_cfg_name,windows_paths)
        #        self.add_entries_to_config_object(major_cfg_name,{"Comment":comment,
        #                                                 "Contact":contact})
   #     else:
            
    file_name=file_name
    with open(file_name,'w') as conf:
        config_object.write(conf)
        print("Config-File ",colored(file_name,"red"), "is created!")

def check_if_config_file_exists(name):
    if not os.path.isfile(name):
        create_new_config_file(file_name=name)
        file_exists=True
    else:
        print("Config-file",name+".ini"," already exists")
    return True

def load_config_file(path,name):
    config_object = ConfigParser()
    file_name=path+"/"+name+".ini"
    print(file_name)
    check_if_config_file_exists(file_name)
    config_object.read(file_name)
    return config_object

def add_entries_to_config_object(config_file_name,entry_dict):
    """

    Parameters
    ----------
    config_file_name: DICT
        file name of the config-file
    entry_dict : DICT
        dictionary of entries to add in the config file.
    Returns
    -------
    None

    """    
    config_object_old= ConfigParser()
    config_object_old.read(config_file_name+".ini")
    
    # add dictionary entries to config_object
    for key in entry_dict.keys():
        config_object_old["Data_Paths"][key]=entry_dict[key]
    config_object_new=config_object_old
    # write config_objects into data_config file    
    with open(config_file_name+".ini",'w') as conf:
        config_object_new.write(conf)
    print("Entries: ",entry_dict.keys(),
          "have added to or changed in the config file")

    return None

def del_entries_from_config_object(entry_dict):
    """

    Parameters
    ----------
    entry_dict : DICT
        dictionary of entries to delete from the config file.
    Returns
    -------
    None.

    """    
    config_object_old= ConfigParser()
    config_object_old.read("data_config.ini")
    
    # dictionary entries to be deleted from config_object
    for key in entry_dict.keys():
        del config_object_old[key]
    
    # write config_objects into data_config file    
    with open("data_config.ini",'w') as conf:
        config_object= ConfigParser()
        config_object.write(conf)
    print("Entries: ",entry_dict.keys(),"have added to the config file")
    return None

def adapt_config_file_to_system(is_windows):
    """
    This function adapts the config_file to
    the system one is working on.

    Parameters
    ----------
    is_windows : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    return None


def print_config_file(major,keys):
    """
    Parameters
    ----------
    keys : DICT.KEYS
    
    Simply returns the desired keys of config_file for quickcheck.    

    Returns
    -------
    None.

    """
    config_file=ConfigParser()
    config_file.read("data_config.ini")
    dictionary=config_file[major][keys]
    print("The defined specifications of the config file for ",
          colored(major,'red'),"[",colored(keys,'green'),"] "
          "are:",colored(dictionary,"magenta"))
    return None

def perform_hamp_default_configuration():
    pass

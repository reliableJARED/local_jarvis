#!/usr/bin/env python3
"""
Dependency Manager for Python Projects
Handles automatic installation and checking of required packages
Enhanced with model download functionality
"""

import subprocess
import sys
import os
import logging
from typing import List
from pathlib import Path

# For unzipping files for windows libvips setup
import zipfile



class Orenda_DependencyManager:
    """
    A class to manage Python package dependencies with automatic installation
    Enhanced with model download capabilities
    """
    
    def __init__(self):
        """
        Initialize the dependency manager
       
        """
        self.logger = logging.getLogger(__name__) # Use module-level logger


    def _unzip_file(self, zip_path: str, extract_to: str) -> bool:
        """
        Unzip a file to the specified directory
        
        Args:
            zip_path (str): Path to the zip file
            extract_to (str): Directory to extract to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.info(f"Extracting {zip_path} to {extract_to}...")
            
            # Create extraction directory if it doesn't exist
            os.makedirs(extract_to, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            
            self.logger.info("Extraction completed successfully!")
            return True
            
        except FileNotFoundError:
            self.logger.error(f"Error: Zip file not found at {zip_path}")
            return False
        except zipfile.BadZipFile:
            self.logger.error(f"Error: Invalid zip file at {zip_path}")
            return False
        except Exception as e:
            self.logger.error(f"Error during extraction: {str(e)}")
            return False
    
    def _add_to_system_path(self, new_path: str) -> bool:
        """
        Add a directory to the system PATH environment variable (Windows)
        
        Args:
            new_path (str): Path to add to system PATH
            
        Returns:
            bool: True if successful, False otherwise
        """
        import winreg
        try:
            # Check if path exists
            if not os.path.exists(new_path):
                self.logger.warning(f"Path {new_path} does not exist!")
                return False
            
            # Open the system environment variables registry key
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                              r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
                              0, winreg.KEY_ALL_ACCESS) as key:
                
                # Get current PATH value
                try:
                    current_path, _ = winreg.QueryValueEx(key, "PATH")
                except FileNotFoundError:
                    current_path = ""
                
                # Check if path is already in PATH
                if new_path.lower() in current_path.lower():
                    self.logger.info(f"Path {new_path} is already in system PATH")
                    return True
                
                # Add new path to PATH
                if current_path and not current_path.endswith(';'):
                    new_system_path = f"{current_path};{new_path}"
                else:
                    new_system_path = f"{current_path}{new_path}"
                
                # Set the new PATH value
                winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, new_system_path)
                
                self.logger.info(f"Successfully added {new_path} to system PATH")
                return True
                
        except PermissionError:
            self.logger.warning("Administrator privileges required to modify system PATH")
            return False
        except Exception as e:
            self.logger.error(f"Error adding to system PATH: {str(e)}")
            return False
    
    def _add_to_user_path(self, new_path: str) -> bool:
        """
        Add a directory to the user PATH environment variable (Windows)
        
        Args:
            new_path (str): Path to add to user PATH
            
        Returns:
            bool: True if successful, False otherwise
        """
        import winreg
        try:
            # Check if path exists
            if not os.path.exists(new_path):
                self.logger.warning(f"Path {new_path} does not exist!")
                return False
            
            # Open the user environment variables registry key
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                              r"Environment",
                              0, winreg.KEY_ALL_ACCESS) as key:
                
                # Get current PATH value
                try:
                    current_path, _ = winreg.QueryValueEx(key, "PATH")
                except FileNotFoundError:
                    current_path = ""
                
                # Check if path is already in PATH
                if new_path.lower() in current_path.lower():
                    self.logger.info(f"Path {new_path} is already in user PATH")
                    return True
                
                # Add new path to PATH
                if current_path and not current_path.endswith(';'):
                    new_user_path = f"{current_path};{new_path}"
                else:
                    new_user_path = f"{current_path}{new_path}"
                
                # Set the new PATH value
                winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, new_user_path)
                
                self.logger.info(f"Successfully added {new_path} to user PATH")
                return True
                
        except Exception as e:
            self.logger.error(f"Error adding to user PATH: {str(e)}")
            return False
    
    def setup_windows_libvips(self) -> bool:
        """
        Setup libvips on Windows by checking, extracting if needed, and adding to PATH.
        Looks for libvips in ../mono-workspace-repo/dependencies/libvips/vips-dev-8.17 relative to current directory.
        
        Returns:
            bool: True if libvips is ready to use, False otherwise
        """
        if os.name != 'nt':  # Only run on Windows
            self.logger.debug("libvips setup is Windows-only")
            return True
        
        """WINDOWS ONLY: Fix libvips issues by adding to PATH
        FIRST: download libvisp here: https://github.com/libvips/build-win64-mxe/releases/download/v8.17.2/vips-dev-w64-web-8.17.2.zip
        Extract to a folder, then update the path below to point to the 'bin' folder
        """

        # Get current directory and construct paths
        current_dir = os.getcwd()
        workspace_dir = os.path.join(current_dir, "dependencies")
        libvips_base = os.path.join(workspace_dir, "libvips")
        bin_path = os.path.join(libvips_base, "vips-dev-8.17", "bin")
        
        self.logger.info("Setting up libvips for Windows...")
        self.logger.debug(f"Looking for libvips at: {bin_path}")
        
        # Check if libvips bin directory exists
        if os.path.exists(bin_path):
            self.logger.info("LibVIPS found, checking PATH...")
            
            # Add to PATH if not already there
            if not self._add_to_system_path(bin_path):
                self.logger.info("Falling back to user PATH...")
                if self._add_to_user_path(bin_path):
                    self.logger.info("LibVIPS added to user PATH. Restart command prompt for changes to take effect.")
                    return True
                else:
                    self.logger.error("Failed to add libvips to PATH")
                    return False
            else:
                self.logger.info("LibVIPS is ready to use")
                return True
        else:
            self.logger.info("LibVIPS not found, looking for zip file to extract...")
            
            # Look for zip file in libvips directory
            if not os.path.exists(libvips_base):
                self.logger.error(f"LibVIPS directory not found: {libvips_base}")
                self.logger.error("Please ensure the libvips zip file is in ../mono-workspace-repo/dependencies/libvips/")
                return False
            
            # Find zip file
            zip_files = [f for f in os.listdir(libvips_base) if f.endswith('.zip') and 'vips' in f.lower()]
            
            if not zip_files:
                self.logger.error("No libvips zip file found in libvips directory")
                self.logger.error(f"Please place the libvips zip file in: {libvips_base}")
                return False
            
            if len(zip_files) > 1:
                self.logger.warning(f"Multiple zip files found, using first one: {zip_files[0]}")
            
            zip_path = os.path.join(libvips_base, zip_files[0])
            self.logger.info(f"Found zip file: {zip_files[0]}")
            
            # Extract the zip file
            if self._unzip_file(zip_path, libvips_base):
                self.logger.info("Extraction successful, checking for bin directory...")
                
                # Check if bin directory now exists
                if os.path.exists(bin_path):
                    self.logger.info("LibVIPS extracted successfully, adding to PATH...")
                    
                    # Add to PATH
                    if not self._add_to_system_path(bin_path):
                        self.logger.info("Falling back to user PATH...")
                        if self._add_to_user_path(bin_path):
                            self.logger.info("LibVIPS setup complete! Restart command prompt for changes to take effect.")
                            return True
                        else:
                            self.logger.error("Failed to add libvips to PATH")
                            return False
                    else:
                        self.logger.info("LibVIPS setup complete!")
                        return True
                else:
                    self.logger.error("LibVIPS bin directory not found after extraction")
                    self.logger.error(f"Expected path: {bin_path}")
                    return False
            else:
                self.logger.error("Failed to extract libvips zip file")
                return False
    

# Example usage
if __name__ == "__main__":
    dep_manager = Orenda_DependencyManager()
    dep_manager.setup_windows_libvips()
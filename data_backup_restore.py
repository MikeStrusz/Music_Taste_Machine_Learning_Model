import streamlit as st
import pandas as pd
import base64
import io
import os
import json
import zipfile
from datetime import datetime

def create_download_link(df, filename, text):
    """
    Create a download link for a dataframe
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def create_zip_download_link(file_dict, zip_filename, text):
    """
    Create a download link for multiple files in a zip
    """
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, df in file_dict.items():
            if isinstance(df, pd.DataFrame):
                csv_data = df.to_csv(index=False).encode()
                zip_file.writestr(filename, csv_data)
    
    zip_buffer.seek(0)
    b64 = base64.b64encode(zip_buffer.getvalue()).decode()
    href = f'<a href="data:application/zip;base64,{b64}" download="{zip_filename}">{text}</a>'
    return href

def data_backup_restore_tab():
    """
    Create a tab for data backup and restore functionality
    """
    st.subheader("Data Backup & Restore")
    
    # Create tabs for backup and restore
    backup_tab, restore_tab = st.tabs(["Backup Data", "Restore Data"])
    
    with backup_tab:
        st.write("Download your data files to back them up. This helps when working with git repositories.")
        
        # List of data files to backup
        data_files = {
            "Album Covers": "data/nmf_album_covers.csv",
            "Album Links": "data/nmf_album_links.csv",
            "Similar Artists": "data/nmf_similar_artists.csv",
            "Liked Similar Artists": "data/liked_artists_only_similar.csv",
            "Personal Feedback": "feedback/feedback.csv",
            "Public Feedback": "feedback/public_feedback.csv",
            "Nuked Albums": "data/nuked_albums.csv"
        }
        
        # Create checkboxes for selecting which files to backup
        st.write("Select which data to backup:")
        selected_files = {}
        for name, path in data_files.items():
            if os.path.exists(path):
                if st.checkbox(f"{name} ({path})", value=True):
                    try:
                        selected_files[name] = pd.read_csv(path, quoting=1)
                    except Exception as e:
                        st.warning(f"Error reading {name}: {e}")
            else:
                st.info(f"{name} file not found at {path}")
        
        # Individual file download buttons
        st.write("### Download Individual Files")
        for name, df in selected_files.items():
            file_path = data_files[name]
            file_name = os.path.basename(file_path)
            st.markdown(
                create_download_link(df, file_name, f"Download {name}"),
                unsafe_allow_html=True
            )
        
        # Download all selected files as zip
        if selected_files:
            st.write("### Download All Selected Files")
            file_dict = {data_files[name].split('/')[-1]: df for name, df in selected_files.items()}
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.markdown(
                create_zip_download_link(
                    file_dict,
                    f"nmf_data_backup_{timestamp}.zip",
                    "Download All Selected Files as ZIP"
                ),
                unsafe_allow_html=True
            )
        
        # Add metadata to help with restoring
        if selected_files:
            metadata = {
                "backup_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "files": list(selected_files.keys()),
                "file_paths": [data_files[name] for name in selected_files.keys()],
                "row_counts": {name: len(df) for name, df in selected_files.items()}
            }
            
            # Display backup metadata
            st.write("### Backup Metadata")
            st.json(metadata)
            
            # Create download link for metadata
            metadata_json = json.dumps(metadata, indent=2)
            b64 = base64.b64encode(metadata_json.encode()).decode()
            st.markdown(
                f'<a href="data:application/json;base64,{b64}" download="backup_metadata_{timestamp}.json">Download Metadata</a>',
                unsafe_allow_html=True
            )
    
    with restore_tab:
        st.write("Upload your backup files to restore your data.")
        
        # File upload for individual CSV files
        st.write("### Restore Individual Files")
        
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file is not None:
            # Try to determine which data file this is
            try:
                df = pd.read_csv(uploaded_file, quoting=1)
                
                # Show preview of the data
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
                
                # Try to guess which file this is
                file_type = "Unknown"
                for name, path in data_files.items():
                    filename = os.path.basename(path)
                    if uploaded_file.name == filename:
                        file_type = name
                        break
                
                # Let user select where to save this
                save_as = st.selectbox(
                    "Save as:",
                    options=list(data_files.keys()),
                    index=list(data_files.keys()).index(file_type) if file_type in data_files else 0
                )
                
                if st.button("Restore This File"):
                    save_path = data_files[save_as]
                    
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    
                    # Save the file
                    df.to_csv(save_path, index=False, quoting=1)
                    st.success(f"Restored {save_as} data to {save_path}")
                    
                    # Clear cache
                    st.cache_data.clear()
                    
                    # Suggest rerunning the app
                    st.info("Please rerun the app to see the restored data.")
                    if st.button("Rerun App"):
                        st.rerun()
            
            except Exception as e:
                st.error(f"Error processing uploaded file: {e}")
        
        # File upload for ZIP backup
        st.write("### Restore from ZIP Backup")
        
        uploaded_zip = st.file_uploader("Upload a ZIP backup file", type="zip")
        if uploaded_zip is not None:
            try:
                # Read the zip file
                zip_buffer = io.BytesIO(uploaded_zip.read())
                
                with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                    # List files in the zip
                    zip_files = zip_ref.namelist()
                    st.write(f"Files in ZIP: {', '.join(zip_files)}")
                    
                    # Create checkboxes for selecting which files to restore
                    st.write("Select which files to restore:")
                    restore_files = {}
                    
                    for zip_file in zip_files:
                        # Try to match with known data files
                        matched = False
                        for name, path in data_files.items():
                            if os.path.basename(path) == zip_file:
                                if st.checkbox(f"Restore {name} ({zip_file})", value=True):
                                    restore_files[zip_file] = path
                                matched = True
                                break
                        
                        if not matched:
                            if st.checkbox(f"Restore {zip_file} (unknown file)", value=False):
                                # Let user select where to save this
                                save_as = st.selectbox(
                                    f"Save {zip_file} as:",
                                    options=list(data_files.keys()),
                                    key=f"save_as_{zip_file}"
                                )
                                restore_files[zip_file] = data_files[save_as]
                    
                    if restore_files and st.button("Restore Selected Files"):
                        restored_count = 0
                        for zip_file, save_path in restore_files.items():
                            try:
                                # Ensure directory exists
                                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                                
                                # Extract and read the file
                                with zip_ref.open(zip_file) as file:
                                    df = pd.read_csv(file, quoting=1)
                                    
                                    # Save the file
                                    df.to_csv(save_path, index=False, quoting=1)
                                    restored_count += 1
                            except Exception as e:
                                st.error(f"Error restoring {zip_file}: {e}")
                        
                        if restored_count > 0:
                            st.success(f"Restored {restored_count} files")
                            
                            # Clear cache
                            st.cache_data.clear()
                            
                            # Suggest rerunning the app
                            st.info("Please rerun the app to see the restored data.")
                            if st.button("Rerun App", key="rerun_after_zip"):
                                st.rerun()
            
            except Exception as e:
                st.error(f"Error processing ZIP file: {e}")

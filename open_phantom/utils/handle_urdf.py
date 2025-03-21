import os
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET


"""
Fixes a specific path that uses the ROS-style 'package://' prefix.
The 'package://' prefix is used in URDF files to refer to files in the same package.
However, when we're not running in a ROS environment, the paths are not valid.
This function tries to find the absolute path of the mesh file.
If the mesh file is not found, the original path is used.
"""
def fix_path(path: str, urdf_dir: str) -> str:
    if path.startswith('package://'):
        parts = path[len('package://'):].split('/', 1)
        if len(parts) == 2:
            package_name, rel_path = parts
            
            # Try potential locations for the mesh
            potential_paths = [
                os.path.join(urdf_dir, rel_path),
                os.path.join(urdf_dir, '../meshes', rel_path),
                os.path.join(urdf_dir, f'../{package_name}', rel_path),
                os.path.join(urdf_dir, '../..', rel_path)
            ]
            
            for possible_path in potential_paths:
                if os.path.exists(possible_path):
                    return possible_path
            
        print(f"Failed to find mesh for package path: {path}")   
         
    return path


"""
Iterates through the URDF file and fixes the paths of all mesh files.
The URDF file is parsed and the mesh paths are modified in-place.
"""
def fix_mesh_paths(urdf_path:str, urdf_dir: str) -> str:
    root = ET.parse(urdf_path).getroot()
    
    try:
        for mesh in root.findall('.//mesh'):
            if 'filename' in mesh.attrib:
                mesh.attrib['filename'] = fix_path(mesh.attrib['filename'], urdf_dir)
    except Exception as e:
        print(f"Error fixing mesh paths: {e}")
        raise e
    
    fixed_path = urdf_path.replace('.urdf', '_fixed.urdf')
        
    return root, fixed_path


"""
Formats the XML tree to be human-readable and writes it to a file.
"""
def format_xml(root: ET.Element, fixed_path: str) -> None:
    xml_str = ET.tostring(root, encoding='utf-8')
    dom = minidom.parseString(xml_str)
    
    with open(fixed_path, 'w', encoding='utf-8') as f:
        # Write with nice indentation but remove extra whitespace
        pretty_xml = dom.toprettyxml(indent='  ')
        # Remove extra blank lines that minidom sometimes adds
        pretty_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])
        f.write(pretty_xml)
        

def handle_urdf(urdf_path: str) -> str:
    # Check if URDF is valid
    if not os.path.exists(urdf_path):
        print(f"Invalid URDF path: {urdf_path}")
        return None
    
    # FIXME: Add check to see if URDF needs to be processed
    
    try:
        urdf_dir = os.path.dirname(urdf_path)
        root, fixed_path = fix_mesh_paths(urdf_path, urdf_dir)  
        format_xml(root, fixed_path)
        print(f"Successfully processed URDF: {fixed_path}")
        return fixed_path
    except Exception as e:
        print(f"Failed to process URDF: {e}")
        raise e
    

if __name__ == '__main__':
    # Example usage
    cwd = os.getcwd()
    urdf_path = os.path.join(cwd, "notebook/phantom/urdf/SO_5DOF_ARM100_05d.SLDASM/urdf/SO_5DOF_ARM100_05d.SLDASM.urdf")  
    handle_urdf(urdf_path)
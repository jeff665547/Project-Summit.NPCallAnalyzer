from pathlib import Path

exe_path = Path(__file__)
utils_dir = exe_path.parent
proj_dir = utils_dir.parent
resource_dir = proj_dir / Path("resource")

arucodb_jfile = resource_dir / Path("aruco_db.json")
chip_jfile = resource_dir / Path("chip.json")
settings_jfile = resource_dir / Path("settings.json")
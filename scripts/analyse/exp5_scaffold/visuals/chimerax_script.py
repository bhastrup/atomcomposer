import sys
import re
from chimerax.core.commands import run


def calc_edge_width(width, height):
    return width / 200


def execute_chimerax_commands(
    session,
    pdb_file_path,
    image_path,
    movie_path,
    atoms_to_deselect,
    selected_action,
    bg_color,
    width,
    height,
    edge_width,
    image_supersample,
    video_supersample,
    video_width,
    video_height,
):
    print("Opening PDB file...")
    run(session, f"open {pdb_file_path}")

    print("Setting style to ball-and-stick...")
    run(session, "style ball")

    print("Coloring atoms by element...")
    run(session, "color byelement")

    print(f"Setting background color to {bg_color}...")
    run(session, f"set bgColor {bg_color}")

    if len(atoms_to_deselect) > 0 and selected_action:
        print("Selecting all atoms...")
        run(session, "select")
        for atom_index in atoms_to_deselect:
            print(f"Deselecting atom with serial number {atom_index}...")
            run(session, f"select subtract /?:1@@serial_number={atom_index}")

        if selected_action == "hide":
            print("Hiding selected atoms...")
            run(session, "hide sel atoms")
            # print("Showing selected atoms...")
            # run(session, "show sel atoms")
        elif selected_action == "highlight":
            print(f"Setting selection highlight width to {edge_width}...")
            run(session, f"graphics selection width {edge_width}")

    perform_random_rotation = False
    if perform_random_rotation:
        run(session, "turn x 90")
        run(session, "turn y 45")
        run(session, "turn z 180")

    print(f"Saving high-resolution image to {image_path}...")
    run(session, f"save {image_path} supersample {image_supersample} width {width} height {height}")

    if movie_path:
        print("Recording movie...")
        run(session, f"movie record supersample {video_supersample} size {video_width},{video_height}")
        run(session, "turn y 2 180")
        run(session, "wait 180")
        run(session, f"movie encode {movie_path}")


def parse_sys_args(sys_argv):
    print(f"Received arguments: {sys_argv}")  # Debugging

    if len(sys_argv) < 2:
        print("❌ Usage: script.py {pdb_path: ..., image_path: ..., movie_path: ..., atoms_to_deselect: [...]} ")
        sys.exit(1)

    # Join everything except the script name into a single string
    raw_args = " ".join(sys_argv[1:])
    print(f"Raw argument string: {raw_args}")  # Debugging

    # Remove the outer curly brackets `{...}`
    if raw_args.startswith("{") and raw_args.endswith("}"):
        raw_args = raw_args[1:-1].strip()

    # Use regex to correctly split key-value pairs **without breaking lists**
    key_value_pairs = re.findall(r"(\w+):\s*(\[.*?\]|\S+)", raw_args)

    args_dict = {}
    for key, value in key_value_pairs:
        key = key.strip()
        value = value.strip()

        # Convert lists manually
        if value.startswith("[") and value.endswith("]"):
            if len(value) > 2:
                value = [int(x.strip()) for x in value[1:-1].split(",")]
            else:
                value = []
        else:
            value = value.rstrip(",")  # ✅ Strip trailing commas from non-list values

        args_dict[key] = value

    for k, v in args_dict.items():
        if v == "null":
            args_dict[k] = None

    print(f"✅ Parsed Arguments: {args_dict}")

    return args_dict


# Call the function with sys.argv
args_dict = parse_sys_args(sys.argv)


pdb_file_path = args_dict["pdb_path"]
image_path = args_dict["image_path"]
movie_path = args_dict["movie_path"]
atoms_to_deselect = args_dict.get("atoms_to_deselect", [])
selected_action = args_dict.get("selected_action", "highlight")
bg_color = args_dict.get("bg_color", "white")

width = height = 1000
image_supersample = 3
edge_width = calc_edge_width(width, height)
video_supersample = 1
video_width = video_height = 800


execute_chimerax_commands(
    session=session,  # noqa: F821 — injected by ChimeraX at runtime
    pdb_file_path=pdb_file_path,
    image_path=image_path,
    movie_path=movie_path,
    atoms_to_deselect=atoms_to_deselect,
    selected_action=selected_action,
    bg_color=bg_color,
    width=width,
    height=height,
    edge_width=edge_width,
    image_supersample=image_supersample,
    video_supersample=video_supersample,
    video_width=video_width,
    video_height=video_height,
)


# exit the script
sys.exit(0)

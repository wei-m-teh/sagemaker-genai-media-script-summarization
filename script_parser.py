def format_text(texts):
    header = None
    formatted_lines = []
    dialog_lines = []
    for idx, line in enumerate(texts):
        line = line[:-1]
        if line.isupper():
            if len(dialog_lines) > 0:
                dialog_lines.insert(0, f"{header.strip()}:")
                formatted_lines.append(dialog_lines)
                dialog_lines = []
            header = line
        else:
            open_parenthesis_idx = line.find("(")
            close_parenthesis_idx = line.find(")")
            if open_parenthesis_idx != -1 and close_parenthesis_idx != -1:
                temp_line = line[:open_parenthesis_idx] + line[close_parenthesis_idx + 1:]
                if temp_line.strip().isupper():
                    if len(dialog_lines) > 0:
                        dialog_lines.insert(0, f"{header.strip()}:")
                        formatted_lines.append(dialog_lines)
                        dialog_lines = []
                    header = line
            else:
                if header:  # This will skip lines that are not part of the plot. e.g. title, or author etc.
                    dialog_lines.append(line.strip())

    dialog_lines.insert(0, f"{header.strip()}:")
    dialog_lines.append(line.strip())  # final line
    formatted_lines.append(dialog_lines)
    return formatted_lines


with open("output.txt") as f:
    lines = f.readlines()
formatted_lines = format_text(lines)
for formatted_line in formatted_lines:
    print(" ".join(formatted_line))




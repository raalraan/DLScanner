#!/usr/bin/env python3
##### Based on the script by Johannes Rosskopp ###########
# import os
# import sys

def spectrum2paramcard(path_spc, mg_output_dir, verbose=0):
    # path_spc = sys.argv[1]
    # path_param_card = os.path.dirname(sys.argv[2])
    # mg_output_dir = os.path.dirname(path_param_card_pre)

    # os.rename(path_param_card, path_param_card + ".old")
    fpc = open(mg_output_dir + "/Cards/param_card_default.dat", "r")
    fout = open(mg_output_dir + "/Cards/param_card.dat", "w")


    def extract_blockname(line):
        tmp = line.split(" ")
        i = 0
        while tmp[i] != "Block":
            i += 1
        i += 1
        while tmp[i] == "":
            i += 1
        return tmp[i]


    def extract_position(line):
        pos = []
        for item in line.split(" "):
            if "Upsilon" in line:
                break
            if "NMSSMTools" in line:
                break
            if "e" in item or "E" in item and "." in item:
                break

            elif item != "" and item != "DECAY":
                pos.append(int(item))
        return pos


    def get_block(blockname):
        block = []
        inblock = False
        blockname = blockname.lower()
        with open(path_spc, "r") as fspc:
            for line in fspc:
                if len(line.lstrip()) <= 0:
                    continue
                elif line.lstrip()[0] == "#":
                    # line is just a comment
                    continue
                elif blockname in line.lower().split(" ") and not (inblock):
                    inblock = True
                    block.append(line)
                    continue
                elif "block" in line.lower() and inblock:
                    return block
                elif inblock and "#" in line and ("E" in line or "e" in line):
                    block.append(line)
                    continue
                elif inblock:
                    return block
        return block


    def write_line(line, value):
        # write coupling value to output file
        tmp = []
        value_written = False
        for item in line.split(" "):
            if not (value_written) and "e" in item or "E" in item and "." in item:
                mantissa, exp = ("{:.7E}".format(value)).split("E")
                tmp.append("{}{}{:+03}".format(mantissa, "e", int(exp)))
                tmp.append(" ")
                value_written = True
            elif item == "":
                tmp.append(" ")
            else:
                if "\n" in item:
                    tmp.append(item)
                else:
                    tmp.append(item)
                    tmp.append(" ")
        # write tmp
        fout.write("".join(tmp))


    def rewrite_block(block, block_input, blockname):
        fout.write(block[0])
        for line in block[1::]:
            # check if line is a comment
            if line.lstrip()[0] == "#":
                fout.write(line)
                continue
            # find the right entry in the given block by comparing the position numbers
            pos = extract_position(line)
            entry_found = False
            for line_input in block_input[1::]:
                if entry_found:
                    break
                if pos == extract_position(line_input):
                    # found the right line now extract the value and write it to the new file
                    for item in line_input.split(" "):
                        if "e" in item or "E" in item and "." in item:
                            write_line(line, float(item))
                            entry_found = True
                            break
            if not (entry_found):
                if verbose > 0:
                    print("WARNING: {} has not been found in spectrum file in Block {}".format(line, blockname))
                fout.write(line)


    def rewrite_decay(line):
        # check if line is a comment
        if line.lstrip()[0] == "#":
            fout.write(line)
            return 0
        # find the right entry in the given block by comparing the position numbers

        pos = extract_position(line)
        entry_found = False
        with open(path_spc, "r") as fspc:
            for line_input in fspc:
                if entry_found:
                    break
                elif len(line_input.lstrip()) <= 0:
                    continue
                elif line_input.lstrip()[0] == "#":
                    # line is just a comment
                    continue
                elif "decay" in line_input.lower().split(" "):
                    if pos == extract_position(line_input):

                        # found the right line now extract the value and write it to the new file
                        for item in line_input.split(" "):
                            if "e" in item or "E" in item and "." in item:
                                write_line(line, float(item))
                                entry_found = True
                                break
        if not (entry_found):
            if verbose > 0:
                print("WARNING: {} has not been found in spectrum file in Block {}".format(line, "DECAY"))
            fout.write(line)


    def readinblock(block):
        if len(block) > 0:
            if "DECAY" in block[0]:
                rewrite_decay(block[0])
                return 0

        name = extract_blockname(block[0])
        block_input = get_block(name)
        if len(block_input) <= 0:
            if verbose > 0:
                print("WARNING: Block {} not found in spectrum file".format(name))
            for line in block:
                fout.write(line)
        elif len(block) <= 0:
            if verbose > 0:
                print("WARNING: Block {} contains no data")
        elif len(block) != len(block_input) and name.lower():
            if verbose > 0:
                print("WARNING: {} in spectrum file does not have the same length as Block {} in param_card.dat".format(
                    block_input[0], name
                ))
            rewrite_block(block, block_input, name)
        else:
            rewrite_block(block, block_input, name)


    block = []
    inblock = False

    for line in fpc:
        if inblock:
            if "Block" in line:
                # you are already in a Block and a new Block starts
                readinblock(block)
                block = []
                block.append(line)
                continue
            elif "DECAY" in line:
                # you are in a block and a DECAY appears, check if it hast to be filled
                readinblock(block)
                block = [line]
                decay_readin = False
                for item in line.split(" "):
                    if ("E" in item or "e" in item) and "." in item:
                        # only fill in values if line contains a number in scientific format
                        readinblock(block)
                        decay_readin = True
                        break
                if not (decay_readin):
                    fout.write(line)
                block = []
                inblock = False
            elif "#" in line and ("E" in line or "e" in line):
                # you are in a block and a line of data that has to be filled in appears
                block.append(line)
                continue
            else:
                # you are in a block and the block ends
                readinblock(block)
                block = []
                fout.write(line)
                inblock = False
                continue
        else:
            if "Block" in line:
                # you are not in a block and a new one opens
                block = []
                block.append(line)
                inblock = True
                continue
            elif "DECAY" in line:
                # you are in a block and a DECAY appears, check if it hast to be filled
                block = [line]
                decay_readin = False
                for item in line.split(" "):
                    if ("E" in item or "e" in item) and "." in item:
                        # only fill in values if line contains a number in scientific format
                        readinblock(block)
                        decay_readin = True
                        break
                if not (decay_readin):
                    fout.write(line)
                block = []
                inblock = False
            else:
                fout.write(line)

    fpc.close()
    fout.close()

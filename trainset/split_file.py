def split_file(file, prefix, max_size, buffer=1024):
	"""
	file: the input file
	prefix: prefix of the output files that will be created
	max_size: maximum size of each created file in bytes
	buffer: buffer size in bytes

	Returns the number of parts created.
	"""
	with open(file, 'r+b') as src:
		suffix = 0
		while True:
			with open(prefix + '.%s' % suffix, 'w+b') as tgt:
				written = 0
				while written < max_size:
					data = src.read(buffer)
					if data:
						tgt.write(data)
						written += buffer
					else:
						return suffix
				suffix += 1


def cat_files(infiles, outfile, buffer=1024):
	"""
	infiles: a list of files
	outfile: the file that will be created
	buffer: buffer size in bytes
	"""
	with open(outfile, 'w+b') as tgt:
		for infile in infiles:
			with open(infile, 'r+b') as src:
				while True:
					data = src.read(buffer)
					if data:
						tgt.write(data)
					else:
						break


for idfile in ["1","4"]:
	#split_file("train_Wash" + idfile + ".mat", "train_Wash" + idfile + "_part", 80*1000*1000)
	cat_files(["train_Wash" + idfile + "_part.0", "train_Wash" + idfile + "_part.1"], "train_Wash" + idfile + ".mat")

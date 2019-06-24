import pandas as pd
import re
import sys

class KW:
	def __init__(self, filename, num_row):
		self._filename_ = filename
		self._num_row_=int(num_row)

	@property
	def filename(self):
		return self._filename_

	@property
	def num_row(self):
		return self._num_row_
	
	def getKWfile(self):
		if self.num_row==0:
			return pd.read_csv(self.filename)
		else:
			df = pd.read_csv(self.filename)
			return df.head(self.num_row)


if __name__ == '__main__':
	kw = KW(sys.argv[1],sys.argv[2])
	print(KW.getKWfile(kw))

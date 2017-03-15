from .. import data
import glob, os

class IMDB(data.TarDataset, data.TabularDataset):
    url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    dirname = 'aclImdb'
    filename = 'aclImdb_v1.tar.gz'

    def _mergeFiles(path, output_file, label):
        filenames = glob.glob(path+'/*')
        print('Merging files in ', path)
        with open(output_file, 'w') as outfile:
            for fname in filenames:
                with open(fname, 'r') as readfile:
                    outfile.write(readfile.read().replace('\t', ' ') + "\t" + label + "\n")
    def _concatFiles(filenames, output_file):
        print('Creating ', output_file)
        with open(output_file, 'w') as outfile:
            for fname in filenames:
                with open(fname) as infile:
                    outfile.write(infile.read())
    @classmethod
    def _createSplitFile(cls,path, extension, out):
        out = os.path.join(path, out)
        path = os.path.join(path, extension)
        neg = os.path.join(path, 'neg')
        pos = os.path.join(path, 'pos')
        cls._mergeFiles(neg, neg + '/tmp' , '0')
        cls._mergeFiles(pos, pos + '/tmp' , '1')
        cls._concatFiles([neg + '/tmp', pos + '/tmp'], out)
        return out

    @classmethod
    def splits(cls, text_field, label_field, root='.', train='imdb.train',
               test='imdb.test', validation=None):
        """Create the large movie review dataset from http://ai.stanford.edu/~amaas/data/sentiment/
        Arguments:
            path: Path to the data file
            text_field: The field that will be used for input text data
            label_field: The field that will be used for output label
        """
        path = cls.download_or_unzip(root)
        imbd_path = os.path.join(root, cls.dirname)

        if(not os.path.isfile(os.path.join(imbd_path, train))):
            print('Processing dataset files')
            cls._createSplitFile(os.path.join(root, cls.dirname),'train', train)
            cls._createSplitFile(os.path.join(root, cls.dirname), 'test', test)
        print('Creating dataset splits')
        return super(IMDB, cls).splits(
            path, train, None, test,
            format='tsv', fields=[('text', text_field), ('label', label_field)])

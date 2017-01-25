from __future__ import print_function

import argparse
import stat
import tempfile
import os
import sys

if sys.version_info[0] == 2: 
    import commands as cmd
else:  # assuming that it is python 3
    import subprocess as cmd


class ArticleFormatter:
    def __init__(self):
        self._init()

    def _init(self):
        """
        Clean the object
        """
        self.index = None
        self.buf = []
        self.has_title = False
        self.has_authors = False
        self.has_abstract = False
        self.has_URL = False
        self.has_notes = False

    def __call__(self, s):
        self.buf.append(s)
        self._analyze()
        if self.has_title and self.has_authors and self.has_abstract and self.has_URL and self.has_notes:
            printed = self._print()
            self._init()
            return printed
        else:
            return ""

    def _analyze(self):
        if not self.has_title:
            if 0 < len(self.buf[-1]) < 100:
                self.has_title = True
        elif not self.has_authors:
            if 0 < len(self.buf[-1]):
                self.has_authors = True
                self.index = len(self.buf)
        elif not self.has_URL:
            if self.buf[-1] and (self.buf[-1][:4] == "http" or self.buf[-1][:8] == "**URL:**"):
                self.has_URL = True
                self.buf = self.buf[:self.index] + [" ".join(filter(lambda x: x, self.buf[self.index:-1])),
                                                    self.buf[-1]]
                self.has_abstract = True
        elif self.buf[-1]:
            self.has_notes = True

    def _print(self):
        self.buf = list(filter(lambda x: x, self.buf))
        if len(self.buf) != 5:
            raise ValueError("Wrong article description format!")
        printed = ""
        if self.buf[0][:3] != "###":
            printed += "### "
        printed += self.buf[0] + "\n\n"
        if self.buf[1][:12] != "**Authors:**":
            printed += "**Authors:** "
        printed += self.buf[1] + "\n\n"
        if self.buf[2][:13] != "**Abstract:**":
            printed += "**Abstract:** "
        printed += self.buf[2] + "\n\n"
        if self.buf[3][:8] != "**URL:**":
            printed += "**URL:** "
        printed += self.buf[3] + "\n\n"
        if self.buf[4][:10] != "**Notes:**":
            printed += "**Notes:** "
        printed += self.buf[4] + "\n\n"

        return printed


parser = argparse.ArgumentParser()
parser.add_argument("--toc-maker", help="path to ToC making tool")

known_args, unknown_args = parser.parse_known_args()

if not known_args.toc_maker:
    known_args.toc_maker = "./gh-md-toc"
    if not os.path.isfile(known_args.toc_maker):
        s = cmd.getoutput("uname -s").lower()
        f = "gh-md-toc.%s.amd64.tgz" % s
        URL = "https://github.com/ekalinin/github-markdown-toc.go/releases/download/0.6.0/%s" % f
        if not os.path.isfile(f):
            if cmd.getstatusoutput("wget %s" % URL)[0] != 0:
                raise EnvironmentError("Cannot download toc maker from URL: %s" % URL)
        if cmd.getstatusoutput("tar xzf %s" % f)[0] != 0:
                raise EnvironmentError("Cannot untar toc maker from file %s" % f)
        os.remove(f)

        current_permissions = stat.S_IMODE(os.lstat(known_args.toc_maker).st_mode)
        os.chmod(known_args.toc_maker, current_permissions & stat.S_IXUSR)

if unknown_args:
    filepath = unknown_args[0]
else:
    print("You should specify the path for file to work with!")
    quit(1)

formatted = ""

with open(filepath) as f:
    pass_lines = False
    formatter = ArticleFormatter()
    for line in f:
        l = line.strip()
        if l == "Table of Contents":
            pass_lines = True
        elif l == "Articles":
            pass_lines = False
            formatted += l + "\n"
        elif l == "========":
            formatted += l + "\n"
        elif l[:5] == "## 20":
            formatted += l + "\n"
        elif not pass_lines:
            formatted += formatter(l)

temp = tempfile.NamedTemporaryFile(delete=False, mode='wt')
temp.write(formatted)
temp.close()
toc = cmd.getoutput("%s %s" % (known_args.toc_maker, temp.name))
os.remove(temp.name)

with open(filepath, "wt") as f:
    f.write(toc[:-74] + formatted)

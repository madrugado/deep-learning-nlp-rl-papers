from __future__ import print_function

import argparse
import stat
import tempfile
import os
import sys
import re
import requests

if sys.version_info[0] == 2: 
    import commands as cmd
    from urllib import urlencode
else:  # assuming that it is python 3
    import subprocess as cmd
    from urllib.parse import urlencode


class ArticleFormatter:
    def __init__(self, twitter_command=""):
        self.twitter_command = twitter_command
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

        self.new_article = False

    def __call__(self, s):
        self.buf.append(s)
        self._analyze()
        if self.has_title and self.has_authors and self.has_abstract and self.has_URL and self.has_notes:
            printed = self._print()
            if self.new_article and self.twitter_command:
                twit = self._twitting()
                print("twitting: " + twit)
            self._init()
            return printed
        else:
            return ""

    def _twitting(self):
        url = shorten_url(self.buf[3] if self.buf[3][:8] != "**URL:**" else self.buf[3][9:])
        text = self.buf[4] if self.buf[4][:10] != "**Notes:**" else self.buf[4][11:]
        if len(text) > 140 - len(url) - 1:  # one symbol for space
            premature_ending = "... "
            # FIXME: for some reason twitter counts for three symbols more, than len()
            while len(text) > 140 - len(premature_ending) - len(url) - 3:
                text = str.rsplit(text, " ", 1)[0]

            twit = "\"" + text + premature_ending + url + "\""
        else:
            twit = "\"" + text + " " + url + "\""

        cmd.getstatusoutput(self.twitter_command + " " + twit)
        return twit

    def _analyze(self):
        # special case of arxiv link
        if not self.has_title and not self.has_authors \
                and self.buf[-1].startswith("http") and "arxiv.org" in self.buf[-1]:
            self.buf = list(parse_arxiv(self.buf[-1])) + [self.buf[-1]]
            self.has_title = True
            self.has_authors = True
            self.has_abstract = True
            self.has_URL = True
            return

        if not self.has_title:
            if 0 < len(self.buf[-1]) < 150:
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
            print("\n".join(self.buf) + "\n")
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
            self.new_article = True
        printed += self.buf[4] + "\n\n"

        return printed


def shorten_url(url):
    resp = requests.get('https://is.gd/create.php?' + urlencode({'url': url, 'format': 'simple'}))
    return resp.text


def parse_arxiv(url):
    resp = requests.get(url).text

    # title
    title_start = resp.find("Title:")
    title_start = resp.find("\n", title_start) + 1
    title = resp[title_start:resp.find("</h1>", title_start)].strip()

    # authors
    authors_start = resp.find("Authors:") + 8
    authors = resp[authors_start:resp.find("</div>", authors_start)]
    authors = re.sub("<[^>]*>", "", authors)
    authors = re.sub("\n", "", authors).strip()

    # abstract
    abstract_start = resp.find("Abstract:")
    abstract_start = resp.find("</span>", abstract_start) + 7
    abstract = resp[abstract_start:resp.find("</blockquote>", abstract_start)]
    abstract = " ".join(abstract.split("\n")).strip()

    # URL in abstract
    url_position_start = abstract.find("<a href=\"")
    if url_position_start >= 0:
        url_position_stop = abstract.find("\"", url_position_start + 9)
        link_stop = abstract.find("</a>", url_position_stop) + 4
        abstract_url = abstract[url_position_start + 9:url_position_stop]
        abstract = abstract[:url_position_start] + "[URL](" + abstract_url + ")" + abstract[link_stop:]
    abstract = re.sub("<[^>]*>", "", abstract)

    return title, authors, abstract


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toc-maker", help="path to ToC making tool")
    parser.add_argument("--twitter-poster", default="t update", help="twitter poster command")
    parser.add_argument("-t", "--use-twitter", action="store_true")

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

    return known_args, filepath


def main():
    known_args, filepath = parse_args()
    formatted = ""

    with open(filepath) as f:
        pass_lines = False
        if known_args.use_twitter:
            formatter = ArticleFormatter(known_args.twitter_poster)
        else:
            formatter = ArticleFormatter()

        for line in f:
            l = line.strip()
            if l == "Table of Contents":
                pass_lines = True
            elif l in ["Articles", "Miscellaneous"]:
                pass_lines = False
                formatted += l + "\n"
            elif l in ["========", "============="]:
                formatted += l + "\n"
            elif l[:3] == "## ":
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


if __name__ == "__main__":
    main()

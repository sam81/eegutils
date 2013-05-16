#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os, platform, time

f = open('../setup.py', 'r')
ln = f.readlines()
f.close()
for i in range(len(ln)):
    if ln[i].strip().split('=')[0].strip() == "version":
           ver = ln[i].strip().split('=')[1].strip()
           ver = ver[1:len(ver)-2]
tarball_path = "../dist/eegutils-" + ver + ".tar.gz"
buldpath = " ../pkg_build/" + platform.linux_distribution[0] + ' ' + platform.linux_distribution[1] + ' ' + platform.uname()[4]

if os.path.exists(os.path.normpath(buildpath)) == False:
    os.makedirs(buildpath)
cmd = "cp " + tarball_path + " " + buildpath
os.system(cmd)

f = open('debian/changelog', 'r')
ln = f.readlines()
f.close()

l0 = ['eegutils (' + ver + ') unstable; urgency=low',
      '\n\n',
      '  * New upstream release',
      '\n\n',
      ' -- Samuele Carcagno <sam.carcagno@gmail.com>  ' + time.strftime("%a, %d %b %Y %H:%M:%S +0100", time.gmtime()),
      '\n\n']
l0.extend(ln)
f = open('debian/changelog', 'w')
ln = f.writelines(l0)
f.close()

os.chdir(buildpath)

cmd2 = "tar -xvf " + "eegutils-" + ver + ".tar.gz"
os.system(cmd2)

cmd3 = "cp -R " + "..prep-release/debian/ ./eegutils-" + ver
os.system(cmd3)


os.chdir("eegutils-" + ver)
os.system("dpkg-buildpackage -F")

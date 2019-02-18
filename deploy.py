import getpass
import optparse
import paramiko
import sys
import os


def stream_pipe(source, target):
  for line in source.readlines():
    target.write(line)

def sftp_putrec(sftp, local, remote, confirm=True, log=True):
  if os.path.isfile(local):
    local = local.replace('./', '')
    remote = remote.replace('./', '')
    print('put %s to %s' % (local, remote)) if log else 0
    return sftp.put(local, remote, confirm=confirm)
  if local.find('.git')>=0:
    return
  if remote!='.':
    print('mkdir %s to %s' % (local, remote)) if log else 0
    try: sftp.mkdir(remote)
    except: pass
  for f in os.listdir(local):
    sftp_putrec(sftp, local+'/'+f, remote+'/'+f, confirm)


p = optparse.OptionParser()
p.set_defaults(host='192.168.43.36', username='otheruser', password='password123', local='', remote='')
p.add_option('--host', dest='host', help='set remote host')
p.add_option('--username', dest='username', help='set login username')
p.add_option('--password', dest='password', help='set login password')
p.add_option('--local', dest='local', help='set local path for copy')
p.add_option('--remote', dest='remote', help='set remote path')
(o, args) = p.parse_args()

ssh = paramiko.SSHClient()
ssh.load_system_host_keys()
ssh.connect(o.host, username=o.username, password=o.password)
print('logged in as %s to %s' % (o.username, o.host))

print('\ncopying files to %s:' % o.host)
if o.local!='' and o.remote!='':
  sftp = ssh.open_sftp()
  sftp_putrec(sftp, o.local, o.remote)
  sftp.close()
  
print('\ninstalling python requirements:')
requirements = o.remote+'/requirements.txt'
stdin, stdout, stderr = ssh.exec_command('/usr/local/bin/pip3 install --user -r '+requirements)
stream_pipe(stdout, sys.stdout)
stream_pipe(stderr, sys.stderr)

print('\nstarting server.py:')
server = o.remote+'/server.py'
stdin, stdout, stderr = ssh.exec_command('cd %s && /usr/local/bin/python3 %s' % (o.remote, server))
stream_pipe(stdout, sys.stdout)
stream_pipe(stderr, sys.stderr)

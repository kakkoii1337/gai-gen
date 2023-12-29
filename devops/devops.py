import os,subprocess, cmd
from termcolor import colored
from dotenv import load_dotenv
load_dotenv()

HELP_TEXT = """
1) build                            ; This will build the docker image only based on one of ['ttt','stt','tts','itt']
2) start                            ; This will build the docker image and start the container based on one of ['ttt','stt','tts','itt']
3) stop                             ; This will stop the container based on one of ['ttt','stt','tts','itt']
4) logs                             ; This iwll show the container logs based on one of ['ttt','stt','tts','itt']
5) ps                               ; This will show the container status
6) push                             ; This will build and push the docker image based on one of ['ttt','stt','tts','itt']
7) publish                          ; This will package distribution and publish to pypi based on one of ['ttt','stt','tts','itt']
"""

class Deploy(cmd.Cmd):

    def _cmd(self,cmd):
        try:
            subprocess.run(cmd,shell=True,check=True)
        except subprocess.CalledProcessError as e:
            print("Error: ",e)

    def _ssh(self,svc):
        self._cmd(f"docker exec -it gai-{svc} bash")

    def do_ssh(self,svc):
        if not svc:
            print("Please specify one from ['ttt','stt','tts','itt'] ")
            return
        self._ssh(svc)

    def do_logs(self,svc):
        if not svc:
            print("Please specify one from ['ttt','stt','tts','itt'] ")
            return
        self._cmd(f"docker logs gai-{svc}")

    def do_exit(self,ignored):
        return True

    def get_version(self):
        with open('../setup.py', 'r') as file:
            first_line = file.readline()
            _, version = first_line.split('=')
            version = version.strip().strip("'")  # remove whitespace and quotes
        return version

    def do_build(self,svc):
        if not svc:
            print("Please specify one from ['ttt','stt','tts','itt'] ")
            return
        version =self.get_version()
        if (svc != "itt"):
            self._cmd("rm -rf working && mkdir working")
            self._cmd("cp ../gai.json working")
            self._cmd("cp -rp ../gai/gen/api working")
            self._cmd(f"docker build -t gai-{svc}:{version} -f Dockerfiles/Dockerfile.{svc.upper()} .")
        else:
            self._cmd("rm -rf working && mkdir working")
            self._cmd("cp ../gai.json working")
            self._cmd("cp -rp ../gai/gen/api working")
            self._cmd("cp -rp ../external/LLaVA working")
            self._cmd(f"docker build -t gai-{svc}:{version} -f Dockerfiles/Dockerfile.{svc.upper()} .")

    def do_start(self,svc):
        if not svc:
            print("Please specify one from ['ttt','stt','tts','itt'] ")
            return
        self.do_stop(svc)
        self.do_build(svc)
        self._cmd(f"""docker run -d \
            --name gai-{svc} \
            -p 12031:12031 \
            -e SWAGGER_URL={os.environ["SWAGGER_URL"]} \
            -e OPENAI_API_KEY={os.environ["OPENAI_API_KEY"]} \
            -e ANTHROPIC_API_KEY={os.environ["ANTHROPIC_API_KEY"]} \
            --gpus all \
            -v ~/gai/models:/app/models \
            gai-{svc}:latest
            """)

    def do_stop(self,svc):
        if not svc:
            print("Please specify one from ['ttt','stt','tts','itt'] ")
            return
        self._cmd(f"""docker rm -f gai-{svc}""")

    def do_push(self,svc):
        if not svc:
            print("Please specify one from ['ttt','stt','tts','itt'] ")
            return
        version=self.get_version()
        self.do_build(svc)
        self._cmd(f"""docker tag gai-{svc}:{version} kakkoii1337/gai-{svc}:{version}""")        
        self._cmd(f"""docker push kakkoii1337/gai-{svc}:{version}""")

    def do_ps(self,ignored):
        self._cmd(f"""docker ps -a""")

    def do_publish(self, svc):
        if not svc:
            print("Please specify one from ['ttt','stt','tts','itt'] ")
            return
        self._cmd("cd ~/github/kakkoii1337/gai-gen && python setup.py sdist")
        self._cmd("cd ~/github/kakkoii1337/gai-gen && twine upload dist/*")        

if __name__ == "__main__":
    Deploy().cmdloop()
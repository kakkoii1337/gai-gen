import os,subprocess, cmd
from termcolor import colored
from dotenv import load_dotenv
load_dotenv()

HELP_TEXT = """
1) build                            ; This will build the docker image only
2) start                            ; This will build the docker image and start the container
3) stop                             ; This will stop the container
4) logs                             ; This iwll show the container logs
5) ps                               ; This will show the container status
6) build_mistral7b_exllama
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
        self._ssh(svc)

    def do_logs(self,svc):
        self._cmd(f"docker logs gai-{svc}")

    def do_exit(self,ignored):
        return True

    def do_build(self,svc):
        if (svc != "itt"):
            self._cmd("rm -rf working && mkdir working")
            self._cmd("cp ../gai.json working")
            self._cmd("cp -rp ../gai/gen/api working")
            self._cmd(f"docker build -t gai-{svc}:latest -f Dockerfiles/Dockerfile.{svc.upper()} .")
        else:
            self._cmd("rm -rf working && mkdir working")
            self._cmd("cp ../gai.json working")
            self._cmd("cp -rp ../gai/gen/api working")
            self._cmd("cp -rp ../external/LLaVA working")
            self._cmd(f"docker build -t gai-{svc}:latest -f Dockerfiles/Dockerfile.{svc.upper()} .")

    def do_start(self,svc):
        self.do_stop(svc)
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
        self._cmd(f"""docker rm -f gai-{svc}""")

    def do_push(self,svc):
        self._cmd(f"""docker tag gai-{svc}:latest kakkoii1337/gai-{svc}:latest""")        
        self._cmd(f"""docker push kakkoii1337/gai-{svc}:latest""")

    def do_ps(self,svc):
        self._cmd(f"""docker ps -a""")

    def do_publish_gai(self, ignored):
        self._cmd("cd ~/github/kakkoii1337/gai-gen && python setup.py sdist")
        self._cmd("cd ~/github/kakkoii1337/gai-gen && twine upload dist/*")        

if __name__ == "__main__":
    Deploy().cmdloop()
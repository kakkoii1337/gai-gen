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
        with open('../gai/gen/api/VERSION', 'r') as file:
            version = file.readline()
        return version

    def do_publish(self, svc):
        if not svc:
            print("Please specify one from ['ttt','stt','tts','itt'] ")
            return
        self._cmd("cd ~/github/kakkoii1337/gai-gen && python setup.py sdist")
        #self._cmd("""cd ~/github/kakkoii1337/gai-gen && twine upload dist/*""")        
        self._cmd("""cd ~/github/kakkoii1337/gai-gen && for file in dist/*; do
                        twine upload "$file" || true
                    done
                  """)        

    def do_build(self,svc):
        if not svc:
            print("Please specify one from ['ttt','stt','tts','itt'] ")
            return

        import threading

        # thread: 1
        def thread_1():
            self.do_publish(svc)

        # thread: 2
        def thread_2():
            version =self.get_version()
            self._cmd("rm -rf working && mkdir working")
            self._cmd("cp ../gai.json working")
            self._cmd("cp ../README.md working")
            self._cmd("cp ../setup.py working")
            self._cmd("cp ../requirements_*.txt working")
            self._cmd("cp -rp ../gai working")
            if (svc == "itt"):
                self._cmd("cp -rp ../external/LLaVA working")
            self._cmd(f"DOCKER_BUILDKIT=1 docker build -t gai-{svc}:{version} -f Dockerfiles/Dockerfile.{svc.upper()} .")
            self._cmd(f"docker tag gai-{svc}:{version} gai-{svc}:latest")

        # Create threads
        t1 = threading.Thread(target=thread_1)
        t2 = threading.Thread(target=thread_2)

        # Start threads
        t1.start()
        t2.start()

        # Wait for both threads to finish
        t1.join()
        t2.join()        

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
        self._cmd(f"""docker tag gai-{svc}:latest kakkoii1337/gai-{svc}:latest""")
        self._cmd(f"""docker push kakkoii1337/gai-{svc}:latest""")

    def do_ps(self,ignored):
        self._cmd(f"""docker ps -a""")


if __name__ == "__main__":
    Deploy().cmdloop()
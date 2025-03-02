"""
this script is used to change the os.envrion

"""
import os

# url, points to where the docker is running.
def setting_url(docker_host):
    if  docker_host == "local_host":
        os.environ[
        "SHOPPING"
        ] = "http://localhost:7770"
        os.environ[
            "SHOPPING_ADMIN"
        ] = "http://localhost:7780/admin"
        os.environ[
            "REDDIT"
        ] = "http://localhost:9999"
        os.environ[
            "GITLAB"
        ] = "http://localhost:8023"
        os.environ[
            "MAP"
        ] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000"
        os.environ[
            "WIKIPEDIA"
        ] = "http://localhost:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
        os.environ[
            "HOMEPAGE"
        ] = "PASS"  # The home page is not currently hosted in the demo site
    elif docker_host == "webarena": # host from webarena's aws docker
        os.environ[
            "SHOPPING"
        ] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7770"
        os.environ[
            "SHOPPING_ADMIN"
        ] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7780/admin"
        os.environ[
            "REDDIT"
        ] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:9999"
        os.environ[
            "GITLAB"
        ] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8023"
        os.environ[
            "MAP"
        ] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000"
        os.environ[
            "WIKIPEDIA"
        ] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
        os.environ[
            "HOMEPAGE"
        ] = "PASS"  # The home page is not currently hosted in the demo site
    else:
        raise ValueError("The docker_host is not supported. Please check the input.")
    print("Done setting up URLs")

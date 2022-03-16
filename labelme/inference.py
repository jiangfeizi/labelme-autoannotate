import pkg_resources as pkg
from subprocess import check_output


class Infer:
    @staticmethod
    def install_requirements(requirements):
        for r in requirements:
            try:
                pkg.require(r)
            except Exception as e:  # DistributionNotFound or VersionConflict if requirements not met
                s = f"{r} not found and is required by YOLOv5"
                print(f"{s}, attempting auto-update...")
                try:
                    print(check_output(f"activate labelme_new && pip install {r}", shell=True).decode())
                except Exception as e:
                    print(f'{e}')
        
    def predict(self, image) -> list:
        print("please write your own Class Infer.")
        return [self.get_shape('test', [[100,100], [200,400]], 'rectangle')]

    def get_shape(self, label, points, shape_type):
        shape = {
            "label": label,
            "points": points,
            "group_id": None,
            "shape_type": shape_type,
            "flags": {}
        }
        return shape


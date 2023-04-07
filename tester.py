import pathvision.core as pathvision
from pathvision.core import PARAMETER_ERROR_MESSAGE, Gradients


def tester():
    gradient_technique = 'Integrated Gradiehynts'
    if not (gradient_technique in iter(Gradients)):
        raise ValueError(
            PARAMETER_ERROR_MESSAGE['INCORRECT_VALUE'].format(gradient_technique, [e for e in Gradients]))


if __name__ == "__main__":
    tester()
# coding: utf-8
import os
import gc
import sys
import numpy
import random as rd
from pysbs import context, sbsenum, sbsarchive, batchtools
from PIL import Image

def get_gpu_engine_for_platform():
    """
    Gets the gpu engine string for the current platform

    :return: string the gpu engine string
    """
    from sys import platform
    if 'linux' in platform:
        return "ogl3"
    elif 'darwin' in platform:
        return 'ogl3'
    elif 'win' in platform:
        return 'd3d10pc'
    raise BaseException("Failed to identify platform")


def param_vec(name_value_pair):
    """
    Generates a string command line parameter string for sbsrender

    :param name_value_pair: name and value of the parameter set as a tuple
    :type name_value_pair: (string, [val])
    :return: string The parameter merged with its value in a batch processor compatible way
    """
    name, value = name_value_pair
    return ('%s@' % name) + ','.join(map(str, value))

def render_maps(material_name, output, params, permutation, sbsar_file, output_size, output_path, use_gpu_engine, output_fmt):
    """
    Invokes sbsrender to render out maps for a material with a set of parameters

    :param material_name: name of the material being rendered
    :type material_name: string
    :param output: name of the output node to be rendered
    :type output: string
    :param params: Instantiated parameters
    :type params: {string: [...]}
    :param permutation: Current permutation
    :type permutation: int
    :param sbsar_file: The sbsar file to render
    :type sbsar_file: string
    :param output_size: the output size for the rendered image. In format 2^n where n is the parameter
    :type output_size: int
    :param output_path: The directory to put the result
    :type output_path: string
    :param use_gpu_engine: Use GPU engine when rendering
    :type use_gpu_engine: bool

    :return: None
    """
    random_number = rd.uniform(0, 10000)
    values = ['$outputsize@%d,%d' % (output_size, output_size),
              '$randomseed@%d' % random_number] + list(map(param_vec, params.items()))
    engine_params = {'engine' : get_gpu_engine_for_platform()} if use_gpu_engine else {}
    batchtools.sbsrender_render(inputs=sbsar_file,
                                output_path=output_path,
                                output_name='{outputNodeName}_%s_%d' % (material_name, permutation),
                                input_graph_output=output,
                                set_value=values,
                                output_format=output_fmt,
                                **(engine_params)).wait()


def calc( input_color, levelinlow, levelinhigh, levelinmid, leveloutlow, levelouthigh ):
    a = min(1, max(0, (numpy.floor(input_color*255)/255 - levelinlow) / ((levelinhigh - levelinlow) if levelinhigh != levelinlow else 1) ))
    b = numpy.abs(levelinmid-0.5)*16+1
    c = numpy.power( a, numpy.power( b, numpy.sign(levelinmid - 0.5) ) ) * (levelouthigh - leveloutlow) + leveloutlow

    return numpy.round( c * 255 - 0.01 )


# main

fp = open('log.txt', 'w')

for _input_color in range(0, 101, 5):
    for _levelinlow in range(0, 6, 5):
        for _levelinhigh in range(0,6,5):
            for _levelinmid in range(0,6,5):
                for _leveloutlow in range(0,6,5):
                    for _levelouthigh in range(0,6,5):

                        if _levelinlow == _levelinhigh:
                            continue

                        input_color = _input_color * 0.01
                        levelinlow = _levelinlow * 0.01
                        levelinhigh = _levelinhigh * 0.01
                        levelinmid = _levelinmid * 0.01
                        leveloutlow = _leveloutlow * 0.01
                        levelouthigh = _levelouthigh * 0.01

                        render_maps(
                            'mat',
                            'basecolor',
                            {
                                'input_color':[input_color],
                                'levelinlow':[levelinlow],
                                'levelinhigh':[levelinhigh],
                                'levelinmid':[levelinmid],
                                'leveloutlow':[leveloutlow],
                                'levelouthigh':[levelouthigh]
                            },
                            0,
                            'Levels_Node.sbsar',
                            16,
                            './',
                            True,
                            'tga'
                        )

                        img = Image.open('basecolor_mat_' + str(0) + '.tga')

                        sd = img.getpixel((0,0))
                        my = calc( input_color, levelinlow, levelinhigh, levelinmid, leveloutlow, levelouthigh )

                        #ファイルをにぎりっぱなしになるのでいちいち削除
                        del img
                        gc.collect()

                        fp.write( format('%s,%f,%f,%f,%f,%f,%f,sd,%d,my,%d,dif,%d\n' % (('OK' if sd==my else 'NG'), input_color, levelinlow, levelinhigh, levelinmid, leveloutlow, levelouthigh, sd, my, my-sd)) )

fp.close()

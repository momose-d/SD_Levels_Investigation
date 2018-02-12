# coding: utf-8
import os
import gc
import sys
import shutil
import numpy
import random as rd
from pysbs import context, sbsenum, sbsarchive, batchtools
from PIL import Image
from multiprocessing import Pool, Value
import multiprocessing as multi

Y_USE_MULTITHREAD   = True      # MultiThread実行する？
Y_USE_PARAM_LIMITED = True     # leveloutlow=0.0, leveouthigh=1.0, levelinmid=0.5 に限定する？
Y_USE_DELTA_UPDATE  = True

Y_OUTPUT_PATH_BASE  = './tmp/'
Y_OUTPUT_PATH_LOG   = Y_OUTPUT_PATH_BASE + 'log/'
Y_OUTPUT_PATH_IMG   = Y_OUTPUT_PATH_BASE + 'img/'
Y_OUTPUT_IMG_EXT    = 'tga'
Y_PARAM_STEP        = 0.05
Y_PARAM_STEP_NUM    = int(1.0 / Y_PARAM_STEP) + 1

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

def render_maps(output, params, sbsar_file, output_path, output_filename, output_size, output_fmt, use_gpu_engine):
    """
    Invokes sbsrender to render out maps for a material with a set of parameters
    :param output: name of the output node to be rendered
    :type output: string
    :param params: Instantiated parameters
    :type params: {string: [...]}
    """
    random_number = rd.uniform(0, 10000)
    values = ['$outputsize@%d,%d' % (output_size, output_size),
              '$randomseed@%d' % random_number] + list(map(param_vec, params.items()))
    engine_params = {'engine' : get_gpu_engine_for_platform()} if use_gpu_engine else {}
    batchtools.sbsrender_render(inputs=sbsar_file,
                                output_path=output_path,
                                output_name='%s' % (output_filename),
                                input_graph_output=output,
                                set_value=values,
                                output_format=output_fmt,
                                **(engine_params)).wait()

def saturate( a ):
    return numpy.clip( a, 0.0, 1.0 )

def calc( _fLevelinlow, _fLevelinhigh, _fLeveloutlow, _fLevelouthigh, _fInput, _fLevelinmid ):
 
    sgn = 1.0 if _fLevelinlow <= _fLevelinhigh else -1.0

    if (_fInput == 0 and _fInput == _fLevelinlow):
        c_0 = 0.0
    elif (_fInput == _fLevelinhigh):
        c_0 = 1.0
    elif (_fInput * sgn) <= (_fLevelinlow * sgn):
        c_0 = 0.0
    elif (_fInput * sgn) >= (_fLevelinhigh * sgn):
        c_0 = 1.0
    else:
        a = saturate( (_fInput - _fLevelinlow) / (_fLevelinhigh - _fLevelinlow) )
        b = numpy.abs(_fLevelinmid - 0.5) * 16 + 1
        c_0 = numpy.power( a, numpy.power( b, numpy.sign(_fLevelinmid - 0.5) ) )

    c = c_0 * (_fLevelouthigh - _fLeveloutlow) + _fLeveloutlow

    return numpy.round( c * 255.0 - 0.01 )
    #return numpy.round( c * 255.0 )
    #return numpy.floor( c * 255.0 )

def thread_func( _uGlobalIdx, _uLocalIdx ):
    strFilename = format( "%04d_%04d" % (_uGlobalIdx, _uLocalIdx) )

    uLocalIdx = _uLocalIdx
    uLevelinlow = uLocalIdx / Y_PARAM_STEP_NUM / Y_PARAM_STEP_NUM
    uLocalIdx -= uLevelinlow * Y_PARAM_STEP_NUM * Y_PARAM_STEP_NUM
    uLevelinhigh = uLocalIdx / Y_PARAM_STEP_NUM
    uLocalIdx -= uLevelinhigh * Y_PARAM_STEP_NUM
    uLeveloutlow = uLocalIdx
    
    strFilePathLog = Y_OUTPUT_PATH_LOG + strFilename + '.txt'

    if Y_USE_DELTA_UPDATE:
       if os.path.exists( strFilePathLog ): return

    fp = open( strFilePathLog , 'w' )
    fp.write( 'levelinlow, levelinhigh, leveloutlow, levelouthigh, input, levelinmid, sd, my, my-sd\n' )

    fLevelinlow  = uLevelinlow  / Y_PARAM_STEP_NUM
    fLevelinhigh = uLevelinhigh / Y_PARAM_STEP_NUM
    fLeveloutlow = uLeveloutlow / Y_PARAM_STEP_NUM

    for uLevelouthigh in range( Y_PARAM_STEP_NUM ):
        fLevelouthigh = uLevelouthigh * Y_PARAM_STEP
        if (Y_USE_PARAM_LIMITED and fLevelouthigh != 1.0): continue

        for uInput in range( Y_PARAM_STEP_NUM ):
            fInput = uInput * Y_PARAM_STEP

            for uLevelinmid in range( Y_PARAM_STEP_NUM ):
                fLevelinmid = uLevelinmid * Y_PARAM_STEP
                if (Y_USE_PARAM_LIMITED and fLevelinmid != 0.5): continue
                
                fInputModified = numpy.floor(fInput*255.0)/255.0

                render_maps(
                    'basecolor',
                    {
                        'input_color':[fInputModified],
                        'levelinlow':[fLevelinlow],
                        'levelinhigh':[fLevelinhigh],
                        'levelinmid':[fLevelinmid],
                        'leveloutlow':[fLeveloutlow],
                        'levelouthigh':[fLevelouthigh]
                    },
                    './Levels_Node.sbsar',
                    Y_OUTPUT_PATH_IMG,
                    strFilename,
                    16,
                    Y_OUTPUT_IMG_EXT,
                    True
                )

                strFilePathImg = Y_OUTPUT_PATH_IMG + strFilename + '.' + Y_OUTPUT_IMG_EXT
                img = Image.open( strFilePathImg )

                sd = img.getpixel((0,0))
                my = calc( fLevelinlow, fLevelinhigh, fLeveloutlow, fLevelouthigh, fInputModified, fLevelinmid )

                # ファイルをにぎりっぱなしになるのでいちいち削除
                del img
                gc.collect()

                # ちょっといったん差分がすぐないのは除外する
                #if abs(sd-my) <2: continue
                
                fp.write(format('%s,%f,%f,%f,%f,%f,%f,sd,%d,my,%d,dif,%d\n' % (('OK' if sd==my else 'NG'), fLevelinlow, fLevelinhigh, fLeveloutlow, fLevelouthigh, fInputModified, fLevelinmid, sd, my, my-sd)) )

    fp.close()

def thread_func_wrapper( _args ):
    thread_func( *_args )

# main
if __name__=='__main__':

    if Y_USE_DELTA_UPDATE == False:
        if os.path.exists( Y_OUTPUT_PATH_BASE ):
            shutil.rmtree( Y_OUTPUT_PATH_BASE )
        os.makedirs( Y_OUTPUT_PATH_LOG )
        os.makedirs( Y_OUTPUT_PATH_IMG )

    uGlobalIdx = 0

    for ulevelinlow in range( Y_PARAM_STEP_NUM ):
        for ulevelinhigh in range( Y_PARAM_STEP_NUM ):
            for uleveloutlow in range( Y_PARAM_STEP_NUM ):
                fLeveloutlow = uleveloutlow * Y_PARAM_STEP
                if (Y_USE_PARAM_LIMITED and fLeveloutlow != 0.0): continue
                
                if Y_USE_MULTITHREAD:
                    p = Pool( multi.cpu_count() )
                    p.map( thread_func_wrapper, [(uGlobalIdx, i) for i in range( Y_PARAM_STEP_NUM * Y_PARAM_STEP_NUM * Y_PARAM_STEP_NUM )] )
                    p.close()
                else:
                    for uLocalIdx in range( Y_PARAM_STEP_NUM * Y_PARAM_STEP_NUM * Y_PARAM_STEP_NUM ):
                        thread_func(uGlobalIdx, uLocalIdx)

                uGlobalIdx += 1

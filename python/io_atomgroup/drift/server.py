def main():
    from python.io_atomgroup.experiments.test import Test
    from drift.model import Model

    import functools
    import logging
    import uvicorn
    # sys.path.append('/app')

    m = Model()
    m.model()

    t = Test(
        # video_args=[],
        url_prefix='/drift',
        feed_fps=30,
        timestamp_font_size=1.5,
        timestamp_offset=(900, 150),
        frame_width=None,
        frame_height=None,
        need_cors=True,
    )
    t.run(
        transform_cb=functools.partial(
            m.process,
        )
    )

    logging.getLogger('ultralytics').setLevel(logging.WARN)
    uvicorn.run(
        t.app,
        port=8081,
        #host='0.0.0.0',
        host='127.0.0.1',
        log_level="info"
    )

if __name__ == '__main__':
    import os
    import sys

    sys.path.append(
        os.path.join(
            os.environ['PROJECT_ROOT'],
            'deps/drift-ml/python/io_atomgroup/',
        ),
    )

    sys.path.append(
        os.environ['PROJECT_ROOT'],
    )

    # import drift
    # import python.io_atomgroup.experiments
    main()

def main():
    from python.io_atomgroup.experiments.test import Test
    from drift.model import Model

    import functools
    import uvicorn
    # sys.path.append('/app')

    m = Model()
    m.model()

    t = Test()
    t.run(
        transform_cb=functools.partial(
            m.process,
        )
    )

    logging.getLogger('ultralytics').setLevel(logging.WARN)
    uvicorn.run(
        t.app,
        port=80,
        host='0.0.0.0',
        log_level="info"
    )

if __name__ == '__main__':
    import os

    sys.path.append(
        os.path.join(
            os.environ['PROJECT_ROOT'],
            'deps/drift-ml/python/io_atomgroup/',
        ),
    )

    sys.path.append(
        PROJECT_ROOT,
    )

    # import drift
    # import python.io_atomgroup.experiments
    main()

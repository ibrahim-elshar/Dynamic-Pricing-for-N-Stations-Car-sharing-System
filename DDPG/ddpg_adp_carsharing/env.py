import sys
import os

# As per: http://stackoverflow.com/a/23386287
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
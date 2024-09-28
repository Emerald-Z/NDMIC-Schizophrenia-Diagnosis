# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

# Set the correct version.
__version__ = "0.5.3"

# Expose .evaluate to the user.
from .evaluation import evaluate

# Expose .explain to the user.
from .functions.explanation_func import explain

# Expose .<function-class>.<function-name> to the user.
from .functions import *

# Expose .<metric> to the user.
from .metrics import *

# Expose .helpers.constants to the user.
from .helpers.constants import *

# Expose the model interfaces.
from .helpers.model import *

# Expose the helpers utils.
from .helpers.utils import *

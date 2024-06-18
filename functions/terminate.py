class AutoExploreTerminateCriteria:
    """
    Terminate criteria for auto explore.
    """

    def __init__(self):
        pass

    def update_status(self, **kwargs):
        """
        Update the status of the terminate criteria.
        """
        pass

    def can_terminate(self) -> bool:
        """
        Returns True if and only if the terminate criteria is met.
        """
        pass

    def describe_criteria(self) -> str:
        """
        Returns a string describing the terminate criteria.
        """
        pass


class AnytimeTerminate(AutoExploreTerminateCriteria):
    """
    Terminate at anytime.
    """

    def can_terminate(self) -> bool:
        return True

    def describe_criteria(self) -> str:
        return "Terminate at anytime."


class IdentifyFileTerminate(AutoExploreTerminateCriteria):
    """
    Terminate when the target file is identified.
    """

    def __init__(self, target_files: list):
        """
        Args:
        - `target_file` (str): The target file.
        """
        self.target_files = target_files.copy()
        self._can_terminate = False

    def update_status(self, **kwargs):
        """
        Args:
        - `kwargs` (dict):
            - 'identified_file' (list): The file identified using 'id'.
        """
        if "identified_file" in kwargs:
            # terminate when the target file is reached
            self._can_terminate = (
                kwargs["identified_file"] in self.target_files)

    def can_terminate(self) -> bool:
        return self._can_terminate

    def describe_criteria(self) -> str:
        return "Terminate when one of the target files is identified."

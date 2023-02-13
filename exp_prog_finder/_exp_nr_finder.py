"""
The code must contain only the data folder containing the corresponding folders 
in "YYMMDD" format. Subsequent sub-folders (with their names in YYMMDD format) 
must contain the data recorded by the C/O monitor system (in *.dat format) .
"""

import calendar, datetime
import pathlib
import requests
import sys

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


class Files:
    """Retrieves information about directories and files inside them to be processed."""

    def __init__(self, path):
        self.directory = path
        self.date = self.directory.stem
        self.file_list = self.get_file_list()
        
        
    def get_file_list(self):
        """
        Returns a list of file names in the directory.
        """
        directory = self.directory.glob("**/*")
        file_list = [x.stem for x in directory if x.is_file()]

        return file_list

class Triggers:

    """Retrieves and processes information about experimental programs from a web API. The API returns information about programs in a JSON format."""

    ARCHIVE_PROGINFO = "http://archive-webapi.ipp-hgw.mpg.de/programs.json?from="

    def __init__(self, date, savefile=False):
        """Initialize the instance of the class.

        Parameters
        ----------
        date : str
            A string in the format "YYMMDD" representing the date for which triggers data will be processed.
        savefile : bool, optional
            Whether the processed data should be saved as a csv file, by default False.
        """

        self.date = date
        self.beginning_of_the_day, self.end_of_the_day = self._convert_to_UTC()
        self.url = self._get_url()
        self.list_of_discharges, self.T0 = self._get_triggers_utc()
        self.df = self._create_df()
        if savefile:
            self.save_file()

    def _convert_to_UTC(self):
        """
        Converts the `date` passed as parameter to UTC timestamps for the start and end of the day.

        Returns
        -------
        tuple
            A tuple of two integers representing the UTC timestamps for the start and end of the day of the `date` passed as parameter.
        """
        year, month, day = (
            int("20" + self.date[:2]),
            int(self.date[2:4]),
            int(self.date[4:]),
        )

        beginning_of_the_day = datetime.datetime(year, month, day, 0, 0, 0, 0)
        finish_of_the_day = datetime.datetime(year, month, day, 23, 59, 59, 0)

        beginning_of_the_day = (
            int(round(calendar.timegm(beginning_of_the_day.timetuple())))
            * 1_000_000_000
            + beginning_of_the_day.microsecond * 1_000
        )

        end_of_the_day = (
            int(round(calendar.timegm(finish_of_the_day.timetuple()))) * 1_000_000_000
            + finish_of_the_day.microsecond * 1_000
        )

        return beginning_of_the_day, end_of_the_day

    def _get_url(self):
        """Returns the URL for accessing triggers data for the `date` passed as parameter.
        """
        url = (
            self.ARCHIVE_PROGINFO
            + str(self.beginning_of_the_day)
            + "&upto="
            + str(self.end_of_the_day)
        )

        return url

    def _get_triggers_utc(self):
        """Returns the discharge numbers and start of each discharge in UTC timestamps for the day of the `date` passed as parameter.
        """
        tab_startow = []
        tab_konca = []
        T0 = []

        wywolanie_dnia = requests.get(self.url)
        number_of_discharges = len(wywolanie_dnia.json()["programs"])
        print(f"Number of discharges on 20{self.date}:", number_of_discharges)

        for discharge_nr in range(number_of_discharges):
            pole_poczatek = wywolanie_dnia.json()["programs"][discharge_nr]["from"]
            tab_startow.append(pole_poczatek)
            pole_koniec = wywolanie_dnia.json()["programs"][discharge_nr]["upto"]
            tab_konca.append(pole_koniec)
            try:
                start_ecrh = wywolanie_dnia.json()["programs"][discharge_nr]["trigger"][
                    "0"
                ][0]
            except (IndexError, TypeError):
                start_ecrh = 0

            T0.append(start_ecrh)

        list_of_discharges = np.arange(1, number_of_discharges + 1)
        return list_of_discharges, T0

    def _create_df(self):
        """Creates a pandas DataFrame from the processed triggers data.

        Returns
        -------
        _type_
            _description_
        """
        new_date_fmt = self.date[:2] + self.date[2:4] + self.date[4:]
        df = pd.DataFrame(columns=["date", "discharge_nr", "T0"])
        df["discharge_nr"] = self.list_of_discharges.astype(int)
        df["T0"] = self.T0
        df["date"] = new_date_fmt
        return df

    def save_file(self):
        """Saves the pandas DataFrame as a csv file with the format "YYMMDD_programs.csv"."""
        self.df.to_csv(f"{self.date}_programs.csv")


class ExpAssignment:
    """This class performs the assignment of experiment numbers to files based on the UTC time they were created.

    This class performs the assignment of experiment numbers to files based on the UTC time they were created.

    Note:
    The class makes use of the Triggers class, pd.DataFrame, and the sys and requests libraries.
    """

    def __init__(self, path, date, file_list, triggers_df, savefile=False):
        """
        Initializes the ExpAssignment object with the directory path and saves the file if specified.

        Parameters:
        -----------
        path: Path
            A Path object representing the directory containing the data files.
        date: int
            A date for which the set of experimental discharges numbers will be assigned.
        file_list: list
            List of files in a given path/directory.
        df: int
            Dataframe of all the discharges performed during given date with together with T0 triggers. 
        savefile: bool, optional
            A flag indicating whether to save the final DataFrame, defaults to False.
        """

        self.date = date
        self.file_list = file_list
        self.triggers_df = triggers_df
        self.total_df = self._make_df()
        self.utc_time = self._get_utc_time()
        self._assign_exp_nr()

        if savefile:
            self.save_file()


    def _make_df(self):
        """
        Returns a DataFrame containing information about all files collected at 
        a given time in the considered directory along with their trigger time 
        T0 and the assigned discharge number.
        """
        # Split file names and process time information
        splitted_fnames = []
        for file_name in self.file_list:
            parts = file_name.split("_")
            if parts[1].startswith("PM"):
                hour = 12 + int(parts[1][2:][:2])
                parts[1] = f"{hour}{parts[1][4:]}"
            elif parts[1].startswith("AM"):
                parts[1] = parts[1][2:]
            splitted_fnames.append(parts)

        # Create the DataFrame containing assigned discharge numbers
        time_data_from_files = [i for i in splitted_fnames]
        df = pd.DataFrame(time_data_from_files)
        df.drop(columns=df.columns[-2], axis=1, inplace=True)
        if len(df.columns) < 3:
            df["type_of_data"] = None
        df.columns = ["date", "time", "type_of_data"]
        df.insert(loc=0, column="file_name", value=self.file_list)

        return df


    def _get_utc_time(self):
        """
        Calculates the UTC timestamp for each row of `total_df` and adds it as a new column named 'utc_time'.

        Returns:
        -------
        List : int
            The list of UTC timestamps for each row of `total_df`.
        """
        self.total_df["utc_time"] = self.total_df.apply(
            lambda row: self._convert_to_UTC(row["date"], row["time"]), axis=1
        )
        self.total_df["discharge_nr"] = "-"

        return self.total_df["utc_time"].tolist()


    def _convert_to_UTC(self, date: str, time: str) -> int:
        """
        Converts a given date and time to UTC timestamp.

        Parameters:
        ----------
        date : str
            A string representation of the date in YYMMDD format.
        time : str
            A string representation of the time in HHMMSS format.

        Returns:
        -------
        int
            The UTC timestamp corresponding to the given date and time.
        """
        date = "20" + date
        discharge_time = datetime.datetime.strptime(f"{date} {time}", "%Y%m%d %H%M%S")
        utc_time = (
            int(round(calendar.timegm(discharge_time.utctimetuple()))) * 1_000_000_000
            + discharge_time.microsecond * 1_000
        )
        return utc_time


    def _assign_exp_nr(self):
        """
        Assigns the discharge number to each data point in `self.total_df` based on its `utc_time`.

        The discharge number is taken from `self.triggers_df`. For each row in `self.total_df`, this function
        iterates through each row in `self.triggers_df` to find a corresponding discharge number, by checking if the
        `utc_time` of the current row in `self.total_df` falls between the `T0` of the current and previous row
        in `self.triggers_df`.

        If a match is found, the discharge number of the corresponding row in `self.triggers_df` is appended to a list.
        The list is then transformed into a numpy array and used to update the `discharge_nr` column in `self.total_df`.

        If there is no discharge registered during the day, a message is printed with the date.
        """
        nr_of_files = []
        for idx_df_total, row_total in self.total_df.iterrows(): 
            for (
                idx_df_triggers,
                row_triggers,
            ) in self.triggers_df.iterrows():
                if idx_df_triggers == 0:
                    continue
                if (
                    self.triggers_df["T0"].loc[idx_df_triggers - 1]
                    < row_total["utc_time"]
                    < self.triggers_df["T0"].loc[idx_df_triggers]
                ):
                    nr_of_files.append([idx_df_total, row_triggers["discharge_nr"]])
        try:
            nr_of_files = np.vstack(nr_of_files)
            self.total_df["discharge_nr"].loc[nr_of_files[:, 0]] = nr_of_files[:, 1]
        except ValueError:
            print(f"\n{self.date} - no discharges registered during the day!\n")

    def save_file(self):
        self.total_df.to_csv(f"{self.date}.csv", sep=";")


def get_all_subdirectories():
    """
    Retrieves all subdirectories in a given directory.
    """
    path = pathlib.Path(__file__).parent.resolve() / "data"
    sub_dirs = [f for f in path.iterdir() if f.is_dir() and f.name[0] != (".")]
    
    return sub_dirs


if __name__ == "__main__":
    list_of_directories = get_all_subdirectories()
    for dir_ in list_of_directories:
        files = Files(dir_)
        date = files.date
        directory = files.directory
        file_list = files.file_list
        if len(file_list)==0:
            continue
        t = Triggers(date)
        df = t.df
        exp_ass = ExpAssignment(directory,date, file_list, df, savefile = True)

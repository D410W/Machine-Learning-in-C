#define NOB_IMPLEMENTATION
#include "nob.h"

#define BUILD_FOLDER "build/"
#define SRC_FOLDER   "src/"

int main(int argc, char **argv)
{
  // This line enables the self-rebuilding. It detects when nob.c is updated and auto rebuilds it then
  // runs it again.
  NOB_GO_REBUILD_URSELF(argc, argv);

  // It's better to keep all the building artifacts in a separate build folder. Let's create it if it
  // does not exist yet.
  //
  // Majority of the nob command return bool which indicates whether operation has failed or not (true -
  // success, false - failure). If the operation returned false you don't need to log anything, the
  // convention is usually that the function logs what happened to itself. Just do
  // `if (!nob_function()) return;`
  if (!nob_mkdir_if_not_exists(BUILD_FOLDER)) return 1;

  // The working horse of nob is the Nob_Cmd structure. It's a Dynamic Array of strings which represent
  // command line that you want to execute.
  Nob_Cmd cmd = {0};

  // nob.h ships with a bunch of nob_cc_* macros that try abstract away the specific compiler.
  // They are verify basic and not particularly flexible, but you can redefine them if you need to
  // or not use them at all and create your own abstraction on top of Nob_Cmd.
  nob_cc(&cmd);
  nob_cc_flags(&cmd);
  nob_cc_output(&cmd, BUILD_FOLDER "mlearning");
  nob_cc_inputs(&cmd, SRC_FOLDER "main.c");
  nob_cmd_append(&cmd, "-lm");
  
#if defined(_WIN32)
  nob_cmd_append(&cmd, "-lBcrypt");
#elif defined(__linux__) || defined(__gnu_linux__)
  // nob_cmd_append(&cmd, "");
#else
    #error "Platform not supported"
#endif // platform specific
  
  if (!nob_cmd_run(&cmd)) return 1;

  return 0;
}

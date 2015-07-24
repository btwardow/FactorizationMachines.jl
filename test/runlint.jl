using Base.Test
using Lint

# lint the code
#@lintpragma( "Ignore dimensionless array field [regv]" )
@test isempty(lintpkg( "FactorizationMachines", returnMsgs=true))


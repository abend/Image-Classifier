

use ExtUtils::MakeMaker;
# See lib/ExtUtils/MakeMaker.pm for details of how to influence
# the contents of the Makefile that is written.
WriteMakefile(
    NAME         => 'Image::Classifier',
    VERSION_FROM => 'lib/Image/Classifier.pm', # finds \$VERSION
    AUTHOR       => 'Sasha Kovar (sasha-cpan@arcocene.org)',
    ABSTRACT     => 'A silhouette-based image classifier.',
    PREREQ_PM    => {
                     'Test::Simple' => 0.44,
                    },
);

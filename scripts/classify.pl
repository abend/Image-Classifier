#!/usr/bin/perl -w
use strict;
use warnings;
use 5.010;

use FindBin qw($Bin);
use lib "$Bin/../lib";
use Image::Classifier;
use Data::Dumper;

my ($tdir, $file, $wdir) = @ARGV;
die "usage: $0 training_dir test_file" unless $tdir && $file;

my $classifier = Image::Classifier->new({training_dir => $tdir,
                                         work_dir => $wdir,
                                         debug_images => 1,
                                         force_refresh => 1});

# my $td = $classifier->{training_data};
# for my $k (keys %$td) {
#   say "$k has ".scalar(@{$$td{$k}})." examples.";
#   my $i = 1;
#   for my $s (@{$$td{$k}}) {
#     say "$k $i has ".@$s." corners";
#     ++$i;
#   }
# }

my ($type, $confidence) = $classifier->classify($file);
say "$file is type '$type' with confidence $confidence";

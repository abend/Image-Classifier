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
                                         debug_images => 3,
                                         force_refresh => 1,
                                         corner_params => {sig=>15, c=>20, t_angle=>175}
                                        });

# my $td = $classifier->{training_data};
# for my $k (keys %$td) {
#   say "$k has ".scalar(@{$$td{$k}})." examples.";
#   my $i = 1;
#   for my $s (@{$$td{$k}}) {
#     my $img = $s->[0];
#     say "$k $img has ".(@$s-1)." corners";
#     ++$i;
#   }
# }

my ($type, $confidence, $match) = $classifier->classify($img);
$type ||= '';
$confidence ||= 0;

say "$file is type '$type' with confidence $confidence.  closest match: $match";
